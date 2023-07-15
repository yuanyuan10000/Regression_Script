import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score,mean_squared_error
from model import DNN
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split


cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#########################################################################################################
def training(x_train, y_train, train_batches=10, batch_size=250):
    model.train()  # 启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
    train_loss = 0
    for batch_idx in range(train_batches):
        data = x_train[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(x_train))].to(device)
        label=y_train[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(x_train))].to(device)
        optimizer.zero_grad()   # 优化器梯度归零
        outputs= model(data)   # 得到输出张量
        loss = F.mse_loss(outputs.view(-1),label.view(-1))   # 计算损失值
        loss.backward()   # 反向传播
        optimizer.step()   # 参数更新
        train_loss += loss.item()   # 计算累计损失
    return train_loss/train_batches   # 返回训练过程的平均损失

def testing(x_test, y_test, test_batches=10,batch_size=250):
    model.eval()  # 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
    test_loss = 0
    y_true_list = np.array([])
    y_score_list = np.array([])
    for batch_idx in range(test_batches):
        data = x_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        label = y_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        outputs= model(data)   # 得到输出张量，即预测值
        loss = F.mse_loss(outputs.view(-1), label.view(-1))   # 计算损失
        test_loss += loss.item()
        y_true_list = np.append(y_true_list, label.detach().cpu().numpy())
        y_score_list = np.append(y_score_list, outputs.view(-1).detach().cpu().numpy())
    return test_loss/test_batches, y_true_list, y_score_list


# 读取数据集
data = pd.read_csv('I:\\1caco2_reg\PHD2_data\RF113_feature_selection_EGLN1_std.csv')
X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train,y_test = np.array(y_train),np.array(y_test)


# train = shuffle(pd.read_csv("PHD2_data/train_Cluster_RF113_feature_selection_EGLN1_std.csv"))
# test = shuffle(pd.read_csv("PHD2_data/test_Cluster_RF113_feature_selection_EGLN1_std.csv"))
# x_train = train.iloc[:,1:-3]
# y_train = np.ravel(train.iloc[:,-3])
# x_test = test.iloc[:,1:-3]
# y_test = np.ravel(test.iloc[:,-3])

x_train=torch.tensor(np.array(x_train)).float()
y_train=torch.tensor(y_train).float()
x_test=torch.tensor(np.array(x_test)).float()
y_test=torch.tensor(y_test).float()

torch.cuda.empty_cache()  # 释放GPU内存
torch.manual_seed(0)   # 设置随机种子

epochs = 500
plot_every = 5   # 每5个epoch记录一下loss
k_fold = 10   # 10折交叉验证
batch_size = 128
per_k = int(x_train.shape[0] / k_fold)   # 每折交叉验证的数据量：140

train_loss_mean = np.empty((k_fold, int(epochs / plot_every)))
test_loss_mean = np.empty((k_fold, int(epochs / plot_every)))
statistics_test = np.empty((int(epochs / plot_every), 1))   # 记录每5个epoch的r2
cross_y_true = np.array([])
cross_y_score = np.array([])

for k in range(k_fold):
    print('##########################{}_fold##############################'.format(k+1))

    # 划分数据
    x_train_k = torch.cat((x_train[:k * per_k], x_train[(k + 1) * per_k:]), dim=0)   # 张量的拼接
    y_train_k = torch.cat((y_train[:k * per_k], y_train[(k + 1) * per_k:]), dim=0)
    x_valid_k = x_train[k * per_k:(k + 1) * per_k]
    y_valid_k = y_train[k * per_k:(k + 1) * per_k]

    # 清除缓存,重新构建一个模型
    torch.cuda.empty_cache()   # 释放显存
    torch.manual_seed(0)   # 为CPU中设置种子
    model = DNN(input_d=113, dropout=0.2, layer1=256, layer2=128).to(device)
    # model.load_state_dict(torch.load('./char-hERGEnsembleRNNwith2d-0-030-0.2741080068916247.pth'))   # 保存模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)   # 优化器


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

    # 训练模型
    for epoch in range(epochs):
        train_loss = training(x_train_k, y_train_k, train_batches=train_batches, batch_size=batch_size)
        test_loss, y_true_list, y_score_list = testing(x_valid_k, y_valid_k,test_batches=valid_batches, batch_size=batch_size)
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
statistics_test = statistics_test / k_fold    # 每折交叉验证的平均r2
statistics_final_cross_Q2 = r2_score(cross_y_true, cross_y_score)
statistics_final_cross_MSE = mean_squared_error(cross_y_true, cross_y_score)

#输出交叉验证的kappa值
statistics_test=pd.DataFrame(statistics_test)
statistics_test.columns = ['R2']
statistics_test.index = [plot_every*(i+1) for i in list(range(int(epochs/plot_every)))]
# statistics_test.to_excel('DNN_RF117_train_R2_each epoch.xlsx',header=True,index=True)

print(statistics_final_cross_Q2,statistics_final_cross_MSE)

######################################### plot_LOSS #####################################
plt.figure(figsize=(10,8))
plt.grid(True)
plt.plot(x_axis,train_loss_mean,'bo',label='Train')
plt.plot(x_axis,train_loss_mean,'b')
plt.plot(x_axis,test_loss_mean,'ro',label='Validation')
plt.plot(x_axis,test_loss_mean,'r')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.title('Training and validation acc',fontsize=25)
plt.legend(fontsize=18)
# plt.savefig('DNN_CV_RF117.png',dpi=300)
plt.show()




