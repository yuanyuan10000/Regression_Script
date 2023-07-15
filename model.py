import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
# import FeaturizerMyOwn
import numpy as np
import torch.distributions as D
import rdkit
from rdkit import Chem
import random



class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]
        
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self):
        
        super(GGNN, self).__init__()
        self.input_dim = 230
        self.state_dim = 64
        self.annotation_dim = 13
        self.n_edge_types = 4
        self.n_node = 60
        self.n_steps = 5
        self.layer1= 128
        self.layer2 = 128
        self.dropout=0.5
        self.mask_null=True
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.possible_atom_types = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's','P', 'F', 'I', 'Cl','Br']  # ZINC
        self.possible_bond_types=np.array([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE])
        

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        #结点信息输出
        self.out1 = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh()
        )

        #注意力机制
        self.out2 = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim,self.state_dim),
            nn.Sigmoid()
        )

        
        #增加二维描述符
        self.out3 =nn.Sequential(
            nn.Linear(self.state_dim + self.input_dim,self.layer1),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer1),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.layer1,self.layer2),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer2),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.layer2,1)
        )

        self._initialization()
                                
    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
        
    def forward(self, prop_state, annotation, A,total_atoms,property_features):
#       prop_state = prop_state.to(torch.float32)
#       annotation = annotation.to(torch.float32)
#       A = A.to(torch.float32)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
                
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)
        
        #节点信息输出
        join_state = torch.cat((prop_state, annotation), 2)        
        x_tanh = self.out1(join_state)
        
        #attention
        x_sigm=self.out2(join_state)
        
        #节点信息乘权重
        graph_feature = (x_sigm*x_tanh).sum(1) 
        #print(graph_feature[0])
        
        '''
        #加上性质参数
        feature=torch.cat((graph_feature,property_features),dim=1)
        output=self.out3(feature)
        '''
        output=None
        return output,graph_feature



#GCN
def normalize_adj(adj):#对边进行标准化
    degrees = np.sum(adj,axis=2)
    # print('degrees',degrees)
    d = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
    for i in range(d.shape[0]):
        d[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
    adj_normal = d@adj@d
    adj_normal[np.isnan(adj_normal)]=0
    return adj_normal



class GCN(nn.Module):
    def __init__(self,
         max_atom=60+8,
         edge_dim=4,
         atom_type_dim=13,
         in_channels=13,
         out_channels=64,
         input_dim=230,
         layer1_dim=64,
         layer2_dim=64,
         dropout=0.5,
         mask_null=True,
         device='cuda' 
        ):
        super(GCN, self).__init__()

        self.max_atom=max_atom
        self.edge_dim=edge_dim
        self.atom_type_dim=atom_type_dim
        self.in_channels=in_channels
        self.out_channels=self.emb_size=out_channels
        self.input_dim = input_dim
        self.layer1= layer1_dim
        self.layer2 = layer2_dim
        self.dropout=dropout
        self.mask_null=mask_null
        self.device=device
          
        #emb layer
        self.Dense0=nn.Linear(self.atom_type_dim,self.in_channels,bias=False)#size(N,H,W,C)
        self.BN0=nn.BatchNorm2d(self.in_channels)#C from an expected input of size (N,C,H,W)
        
        #GCN1
        v1=torch.FloatTensor(1,self.edge_dim, self.in_channels, self.out_channels)
        nn.init.xavier_uniform_(v1)        
        self.v1=nn.Parameter(v1)
        self.BN1=nn.BatchNorm2d(self.emb_size)#C from an expected input of size (N,C,H,W)
        
        #GCN2
        v2=torch.FloatTensor(1,self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v2)        
        self.v2=nn.Parameter(v2)
        self.BN2=nn.BatchNorm2d(self.emb_size)#C from an expected input of size (N,C,H,W)
        
        #GCN3
        v3=torch.FloatTensor(1,self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v3)        
        self.v3=nn.Parameter(v3)
        self.BN3=nn.BatchNorm2d(self.emb_size)#C from an expected input of size (N,C,H,W)
 
        #Linear
        self.linear4=nn.Linear(self.emb_size,self.emb_size,bias=False)
        self.BN4=nn.BatchNorm2d(self.emb_size)#C from an expected input of size (N,C,H,W)

        
        #增加二维描述符
        self.linear5 =nn.Sequential(
            nn.Linear(self.emb_size + self.input_dim,self.layer1),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer1),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.layer1,self.layer2),
            nn.ReLU(),
            nn.BatchNorm1d(self.layer2),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.layer2,1)
        )


    def forward(self, x,property_features):#输入state 即observation
        adj=x['adj']
        node=x['node']
        
        for i in range(adj.shape[0]):
            adj[i]=normalize_adj(adj[i])

        adj=torch.tensor(adj).float().to(self.device)#必须是浮点数        
        node=torch.tensor(node).float().to(self.device)  
        
        #GCN层
        emb_node=self.GCN_Mutilayer(adj,node)
        
        #linear
        emb_node=F.relu(self.linear4(emb_node))
        emb_node=emb_node.permute(0,3,1,2)#把最后一维提到第二维
        emb_node = self.BN4(emb_node)
        emb_node = emb_node.permute(0,2,3,1)#再把第二维放到最后一维
        emb_node = torch.sum(emb_node.squeeze(1), dim=1)  # B*f
        
        #linear
        feature=torch.cat((emb_node,property_features),dim=1)
        pred =self.linear5(feature)
        return pred

    def GCN_Mutilayer(self,adj,node): 
        #emb layer
        ob_node=self.Dense0(node)        
        ob_node=ob_node.permute(0,3,1,2)#把最后一维提到第二维
        ob_node = self.BN0(ob_node)
        ob_node=ob_node.permute(0,2,3,1)#再把第二维放到最后一维

        #GCN1             
        emb_node = self.GCN_batch(adj, ob_node, self.emb_size,self.v1,aggregate='mean')#维度变成（batch_size,1,max_atom,emb_size）
        emb_node=emb_node.permute(0,3,1,2)#把最后一维提到第二维
        emb_node = self.BN1(emb_node)
        emb_node = emb_node.permute(0,2,3,1)#再把第二维放到最后一维
                
        #GCN2
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size,self.v2,aggregate='mean')#维度变成（batch_size,1,max_atom,emb_size）
        emb_node=emb_node.permute(0,3,1,2)#把最后一维提到第二维
        emb_node = self.BN2(emb_node)
        emb_node = emb_node.permute(0,2,3,1)#再把第二维放到最后一维
                
        #GCN3
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size,self.v3,is_act=False,aggregate='mean')#维度变成（batch_size,1,max_atom,emb_size）
        emb_node=emb_node.permute(0,3,1,2)#把最后一维提到第二维
        emb_node = self.BN3(emb_node)
        emb_node = emb_node.permute(0,2,3,1)#再把第二维放到最后一维
        
        #去掉维度为1的那一维数据             
        #emb_node = emb_node.squeeze(1)  # B*n*f
        return emb_node
    
    # gcn mean aggregation over edge features
    def GCN_batch(self,adj, node_feature, out_channels,weight,is_act=True,is_normalize=False, name='gcn_simple',aggregate='sum'):
        '''
        state s: (adj,node_feature)
        :param adj: none*b*n*n
        :param node_feature: none*1*n*d
        :param out_channels: scalar
        :param name:
        :return:
        '''  
        node_embedding = adj@node_feature.repeat(1,self.edge_dim,1,1)@weight.repeat(node_feature.size(0),1,1,1)

        if is_act:
            node_embedding = F.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = torch.sum(node_embedding, dim=1, keepdim=True)  # sum pooling
        elif aggregate=='mean':
            node_embedding = torch.mean(node_embedding,dim=1,keepdim=True) # mean pooling
        elif aggregate=='concat':
            node_embedding = torch.concat(torch.split(node_embedding,self.edge_dim,dim=1),dim=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = F.normalize(node_embedding,p=2,dim=-1)#l2正则化
        
        return node_embedding
    


class DNN(nn.Module):
    def __init__(self, input_d=23, dropout=0.5,layer1=64, layer2=32, layer3=1, Cuda=True):
        super(DNN, self).__init__()
        self.input_d = input_d
        self.Cuda=Cuda
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_d, self.layer1)
        self.BN1 = nn.BatchNorm1d(self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.BN2 = nn.BatchNorm1d(self.layer2)   # 批标准化
        self.fc3 = nn.Linear(self.layer2,self.layer3)

    def forward(self,x):
        x = self.BN1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.BN2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x= self.fc3(x)
        return x
 
        
        
