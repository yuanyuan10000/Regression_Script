import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import roc_curve, auc
from model import DNN
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import math


def calculate_k_kR(y_true, y_score):
    a = (y_true * y_score).sum()
    b = (y_score ** 2).sum()
    c = (y_true ** 2).sum()
    k = a / b
    kR = a / c
    return k, kR


def calculate_r02_r02R(y_true, y_score, k, kR):
    a = ((y_true - k * y_score) ** 2).sum()
    b = ((y_true - y_true.mean()) ** 2).sum()
    r02 = 1 - a / b
    c = ((y_score - kR * y_true) ** 2).sum()
    d = ((y_score - y_score.mean()) ** 2).sum()
    r02R = 1 - c / d
    return r02, r02R


def calculate_rm2_rm2R(r2, r02, r02R):
    rm2 = r2 * (1 - math.sqrt(r2 - r02))
    rm2R = r2 * (1 - math.sqrt(r2 - r02R))
    return rm2, rm2R


def calculate_r2(y_true, y_score):
    y_true_res = y_true - y_true.mean()
    y_score_res = y_score - y_score.mean()
    a = (y_true_res * y_score_res).sum()
    b = (y_true_res ** 2).sum()
    c = (y_score_res ** 2).sum()
    d = math.sqrt(b * c)
    r2 = (a / d) ** 2
    return r2


def CalculateMetrics(y_true, y_score):
    r2 = r2_score(y_true, y_score)
    mse = mean_squared_error(y_true, y_score)
    k, kR = calculate_k_kR(y_true, y_score)
    r02, r02R = calculate_r02_r02R(y_true, y_score, k, kR)
    r2 = calculate_r2(y_true, y_score)
    rm2, rm2R = calculate_rm2_rm2R(r2, r02, r02R)
    data = np.array([r2, mse, k, kR, r02, r02R, r2, rm2, rm2R])
    return data


def CalculateMetricsClassify(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    acc = accuracy_score(y_true, y_score)
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)
    kappa = cohen_kappa_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_score)
    cm = confusion_matrix(y_true, y_score)  ###cm[1,1]第一个1代表真实值为1 第二个1代表预测值为1
    print(cm)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    BAC = (sensitivity + specificity) / 2
    data = np.array(
        [cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0], roc_auc, acc, precision, recall, BAC, f1, kappa, mcc, sensitivity,
         specificity])
    data = data.reshape(1, -1)
    return data


############数据集读取##########################
np.random.seed(0)
dataset = 'AKT1-IC50-single-protein'

x_train = pd.read_csv('./shuffle data/' + dataset + ' training set_RF_selected.csv', header=None)
# X_train_ecfp=pd.read_csv('./shuffle data/training set_ECFP4.csv',header=None)
# X_train_ecfp=X_train_ecfp.iloc[:,1:]
# X_train=pd.concat([X_train,X_train_ecfp],axis=1)
y_train = pd.read_csv('./shuffle data/' + dataset + ' training set_label.csv', header=None)
y_train = np.ravel(y_train.iloc[:, 0])

x_test = pd.read_csv('./shuffle data/' + dataset + ' test set_RF_selected.csv', header=None)
# X_test_ecfp=pd.read_csv('test set_ECFP4.csv',header=None)
# X_test_ecfp=X_test_ecfp.iloc[:,1:]
# X_test=pd.concat([X_test,X_test_ecfp],axis=1)
y_test = pd.read_csv('./shuffle data/' + dataset + ' test set_label.csv', header=None)
y_test = np.ravel(y_test.iloc[:, 0])

# 读取外部验证
x_external = pd.read_csv('./shuffle data/' + dataset + ' external set_RF_selected.csv', header=None)
y_external = pd.read_csv('./shuffle data/' + dataset + ' external set_label.csv', header=None)
y_external = np.ravel(y_external.iloc[:, 0])

x_train = torch.tensor(np.array(x_train)).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(np.array(x_test)).float()
y_test = torch.tensor(y_test).float()
x_external = torch.tensor(np.array(x_external)).float()
y_external = torch.tensor(y_external).float()


# 训练
def training(epoch, x_train, y_train, train_batches=10, batch_size=250):
    model.train()
    train_loss = 0
    for batch_idx in range(train_batches):
        optimizer.zero_grad()
        data = x_train[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(x_train))].to(device)
        label = y_train[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(x_train))].to(device)
        outputs = model(data)
        loss = F.mse_loss(outputs.view(-1), label.view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if batch_idx % 50 == 0:
        # print(f'{epoch} / {batch_idx}\t{loss:.4f}')
    return train_loss / train_batches


def testing(epoch, x_test, y_test, test_batches=10, batch_size=250):
    model.eval()
    test_loss = 0
    y_true_list = np.array([])
    y_score_list = np.array([])
    for batch_idx in range(test_batches):
        data = x_test[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(x_test))].to(device)
        label = y_test[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(x_test))].to(device)
        outputs = model(data)
        loss = F.mse_loss(outputs.view(-1), label.view(-1))
        test_loss += loss.item()
        y_true_list = np.append(y_true_list, label.detach().cpu().numpy())
        y_score_list = np.append(y_score_list, outputs.view(-1).detach().cpu().numpy())
    return test_loss / test_batches, y_true_list, y_score_list


##################使用交叉验证########################

# 使用cudnn
torch.manual_seed(0)
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 120
plot_every = 5
k_fold = 10
batch_size = 128
per_k = int(x_train.shape[0] / k_fold)

# 五折交叉验证#######
train_loss_mean = np.empty((k_fold, int(epochs / plot_every)))
test_loss_mean = np.empty((k_fold, int(epochs / plot_every)))
statistics_test = np.empty((int(epochs / plot_every), 1))
cross_y_true = np.array([])
cross_y_score = np.array([])

# 训练集又从中随机划分出验证集,做交叉验证
for k in range(k_fold):
    print('k_fold', k)
    print('##########')

    # 清除缓存,重新构建一个模型
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    model = DNN(input_d=431, dropout=0.5, layer1=256, layer2=256).to(device)
    # model.load_state_dict(torch.load('./char-hERGEnsembleRNNwith2d-0-030-0.2741080068916247.pth'))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 划分数据
    x_train_k = torch.cat((x_train[:k * per_k], x_train[(k + 1) * per_k:]), dim=0)
    y_train_k = torch.cat((y_train[:k * per_k], y_train[(k + 1) * per_k:]), dim=0)
    x_valid_k = x_train[k * per_k:(k + 1) * per_k]
    y_valid_k = y_train[k * per_k:(k + 1) * per_k]

    # batch
    if len(x_train_k) % batch_size == 0:
        train_batches = int(len(x_train_k) / batch_size)
    else:
        train_batches = int(len(x_train_k) / batch_size) + 1

    if len(x_valid_k) % batch_size == 0:
        valid_batches = int(len(x_valid_k) / batch_size)
    else:
        valid_batches = int(len(x_valid_k) / batch_size) + 1

    x_axis = []
    train_loss_list = []
    test_loss_list = []
    train_loss_avg = 0
    test_loss_avg = 0
    statistic_test_k = np.empty((int(epochs / plot_every), 1))

    for epoch in range(1, epochs + 1):
        train_loss = training(epoch, x_train_k, y_train_k, train_batches=train_batches, batch_size=batch_size)
        test_loss, y_true_list, y_score_list = testing(epoch, x_valid_k, y_valid_k,
                                                       test_batches=valid_batches, batch_size=batch_size)
        train_loss_avg += train_loss
        test_loss_avg += test_loss

        if epoch % plot_every == 0:
            print('epoch', epoch)
            print('train_loss', train_loss_avg / plot_every)
            print('test_loss', test_loss_avg / plot_every)
            train_loss_list.append(train_loss_avg / plot_every)
            test_loss_list.append(test_loss_avg / plot_every)
            train_loss_avg = 0
            test_loss_avg = 0
            statistic_test_k[int(epoch / plot_every) - 1, :] = r2_score(y_true_list, y_score_list)
            x_axis.append(epoch)

    train_loss_mean[k, :] = np.array(train_loss_list)
    test_loss_mean[k, :] = np.array(test_loss_list)
    statistics_test += statistic_test_k
    cross_y_true = np.append(cross_y_true, y_true_list)
    cross_y_score = np.append(cross_y_score, y_score_list)

train_loss_mean = np.mean(train_loss_mean, axis=0)
test_loss_mean = np.mean(test_loss_mean, axis=0)
statistics_test = statistics_test / k_fold
statistics_final_cross_Q2 = r2_score(cross_y_true, cross_y_score)
statistics_final_cross_MSE = mean_squared_error(cross_y_true, cross_y_score)

##LOSS
plt.figure(figsize=(10, 8))
plt.grid(True)
plt.plot(x_axis, train_loss_mean, 'bo', label='Train')
plt.plot(x_axis, train_loss_mean, 'b')
plt.plot(x_axis, test_loss_mean, 'ro', label='Validation')
plt.plot(x_axis, test_loss_mean, 'r')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
# plt.title('Training and validation acc',fontsize=25)
plt.legend(fontsize=18)
plt.savefig('./result/' + dataset + ' DNN_RF.png', dpi=200)
plt.show()

# 输出交叉验证的kappa值
statistics_test = pd.DataFrame(statistics_test)
statistics_test.columns = ['R2']
statistics_test.to_excel('./result/' + dataset + ' DNN_RF_statistics_train_each epoch.xlsx', header=True, index=False)

# 输出交叉验证的最后一轮训练的r2值
# statistics_final_cross=pd.DataFrame(statistics_final_cross)
# statistics_final_cross=statistics_final_cross.T
# statistics_final_cross.columns = ['R2']
# statistics_final_cross.to_excel('./result/'+dataset+' DNN_RF_statistics_train.xlsx',header=True,index=False)

print(statistics_final_cross_Q2, statistics_final_cross_MSE)
# 0.6867527675198777 0.5193802719066964


'''
##############五折交叉验证之后清除缓存,重新构建一个模型,并进行预测###################
#训练
def training(epoch,x_train,y_train,train_batches=10,batch_size=250):
    model.train()
    train_loss = 0
    for batch_idx in range(train_batches):
        optimizer.zero_grad()
        data = x_train[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_train))].to(device)
        label=y_train[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_train))].to(device)
        outputs= model(data)
        loss = F.mse_loss(outputs.view(-1),label.view(-1))
        train_loss += float(loss)
        loss.backward()   
        optimizer.step()
    return train_loss / train_batches

def testing(x_test,y_test,test_batches=10,batch_size=250):
    model.eval()
    test_loss = 0
    y_true_list=np.array([])
    y_score_list=np.array([])
    for batch_idx in range(test_batches):
        data = x_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        label=y_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        outputs= model(data)

        y_true_list=np.append(y_true_list,label.detach().cpu().numpy())
        y_score_list=np.append(y_score_list,outputs.view(-1).detach().cpu().numpy())
    statistic = CalculateMetrics(y_true_list,y_score_list)
    return test_loss / test_batches,statistic,np.ravel(y_true_list),np.ravel(y_score_list)


###########训练模型#############
#使用cudnn
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.empty_cache() 
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs =120
batch_size=128
cell=256


model = DNN(input_d=431,dropout=0.5,layer1=cell,layer2=cell).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.01)


#batch
if len(x_train)%batch_size ==0:
    train_batches=int(len(x_train)/batch_size)
else:
    train_batches=int(len(x_train)/batch_size)+1

if len(x_test)%batch_size ==0:
    test_batches=int(len(x_test)/batch_size)
else:
    test_batches=int(len(x_test)/batch_size)+1

if len(x_external)%batch_size ==0:
    external_batches=int(len(x_external)/batch_size)
else:
    external_batches=int(len(x_external)/batch_size)+1



#train
for epoch in range(1, epochs + 1):        
    loss= training(epoch,x_train,y_train,train_batches=train_batches,batch_size=batch_size)
    if epoch%10==0:
        print(epoch,loss)


torch.save(model.state_dict(), './result/'+dataset+' DNN_RF.pth')


####result###
torch.manual_seed(0)
torch.cuda.empty_cache()  
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DNN(input_d=431,dropout=0.5,layer1=cell,layer2=cell).to(device)
#optimizer = optim.Adam(model.parameters(),lr=0.001)
model.load_state_dict(torch.load('./result/'+dataset+' DNN_RF.pth'))
model.eval()

_,statistics_train,y_train_list,y_train_predicted= testing(x_train,y_train,test_batches=train_batches,batch_size=batch_size)
_,statistics_test,y_test_list,y_test_predicted=testing(x_test,y_test,test_batches=test_batches,batch_size=batch_size)
_,_,y_external_list,y_external_predicted=testing(x_external,y_external,test_batches=external_batches,batch_size=batch_size)


#输出结果
statistics=pd.DataFrame(np.vstack((statistics_train,statistics_test)))
statistics.columns = ['R2','MSE','k','k\'','r02','r0\'2','r2','rm2','rm\'2']
statistics.index=['Train','Test']
statistics.to_excel('./result/'+dataset+' DNN_RF_statistics.xlsx',header=True,index=False)


#测试集数据输出
res=abs(y_test_list-y_test_predicted)
test_out=pd.DataFrame(np.hstack((y_test.reshape(-1,1),y_test_predicted.reshape(-1,1),res.reshape(-1,1))))
test_out.columns = ['y_test_true','y_test_score','res']
test_out.to_excel('./result/'+dataset+' DNN_RF_test_result.xlsx',header=True,index=True)

#训练集数据输出
res=abs(y_train_list-y_train_predicted)
train_out=pd.DataFrame(np.hstack((y_train.reshape(-1,1),y_train_predicted.reshape(-1,1),res.reshape(-1,1))))
train_out.columns = ['y_train_true','y_train_score','res']
train_out.to_excel('./result/'+dataset+' DNN_RF_train_result.xlsx',header=True,index=True)


#外部验证集结果
y_external_predicted=pd.DataFrame(y_external_predicted)
y_external_predicted.columns=['Predicted']
y_external_predicted.to_excel(r'./result/'+dataset+' DNN_RF_predicted.xlsx',index=False)


y_external_predicted[y_external_predicted < 6]=0
y_external_predicted[y_external_predicted >= 6]=1
statistics_external = CalculateMetricsClassify(y_external_list,y_external_predicted)
statistics_external=pd.DataFrame(statistics_external)
statistics_external.columns = ['TP','FP','FN','TN','AUC','ACC','Precision','Recall','BAC','F1','kappa','MCC','Sensitivity','Specificity']
#statistics_external.to_excel(r'./result/RF_RF_exteral_result.xlsx',header=True,index=False)

print('Test',statistics.loc['Test','R2'])
print('external BAC',statistics_external.loc[0,'BAC'])
'''

'''
lw = 2
plt.figure(figsize=(10,8))

plt.grid(True)
plt.plot([-6.5,1.5],[-6.5,1.5],color='black',lw=2)
plt.scatter(y_train,y_train_predicted,s=40,color='#FF9966',marker='^',label='R2-Train= '+str(round(statistics_train[0],3)))
plt.scatter(y_test,y_test_predicted,s=40,color='#336699',marker='v',label='R2-Test = '+str(round(statistics_test[0],3)))
plt.xlim(-6.5,1.5)
plt.ylim(-6.5,1.5)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Experimental Values(-logIC50)',fontsize=25)
plt.ylabel('Predicted Values(-logIC50)',fontsize=25)
plt.legend(loc="lower right",fontsize=20)
plt.savefig('./result/'+dataset+'-DNN_RF.png',DPI=200)
plt.show()
'''