import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
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
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    R2 = r2_score(y_true, y_score)
    mse = mean_squared_error(y_true, y_score)
    k, kR = calculate_k_kR(y_true, y_score)
    r02, r02R = calculate_r02_r02R(y_true, y_score, k, kR)
    r2 = calculate_r2(y_true, y_score)
    rm2, rm2R = calculate_rm2_rm2R(r2, r02, r02R)
    data = np.array([R2, mse, k, kR, r02, r02R, r2, rm2, rm2R])
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

#训练
def training(x_train,y_train,train_batches=10,batch_size=250):
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
    y_true_list = np.array([])
    y_score_list = np.array([])
    for batch_idx in range(test_batches):
        data = x_test[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(x_test))].to(device)
        label = y_test[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(x_test))].to(device)
        outputs = model(data)

        y_true_list=np.append(y_true_list, label.detach().cpu().numpy())
        y_score_list=np.append(y_score_list, outputs.view(-1).detach().cpu().numpy())
    statistic = CalculateMetrics(y_true_list, y_score_list)
    return test_loss / test_batches, statistic, np.ravel(y_true_list), np.ravel(y_score_list)


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
batch_size=128

#batch
if len(x_train)%batch_size ==0:
    train_batches=int(len(x_train)/batch_size)
else:
    train_batches=int(len(x_train)/batch_size)+1

if len(x_test)%batch_size ==0:
    test_batches=int(len(x_test)/batch_size)
else:
    test_batches=int(len(x_test)/batch_size)+1


model = DNN(input_d=113, dropout=0.2, layer1=256, layer2=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#train
for epoch in range(1, epochs + 1):
    loss = training(x_train, y_train, train_batches=train_batches, batch_size=batch_size)
    if epoch%10 == 0:
        print(epoch, loss)


torch.save(model.state_dict(), './PHD2_result/DNN_PHD2.pth')


####################################### result #################################
model.load_state_dict(torch.load('./PHD2_result/DNN_PHD2.pth'))

_, statistics_train, y_train_list, pred_train = testing(x_train, y_train, test_batches=train_batches, batch_size=batch_size)
_, statistics_test, y_test_list, pred_test = testing(x_test, y_test, test_batches=test_batches, batch_size=batch_size)

#输出结果
statistics=pd.DataFrame(np.vstack((statistics_train, statistics_test)))
statistics.columns = ['R2', 'MSE', 'k', 'k\'', 'r02', 'r0\'2', 'r2', 'rm2', 'rm\'2']
statistics.index=['Train', 'Test']
statistics.to_excel('./PHD2_result/DNN_PHD2_RF113_statistics.xlsx',header=True,index=True)

#训练集数据输出
res=abs(y_train_list-pred_train)
train_out=pd.DataFrame(np.hstack((y_train.reshape(-1,1),pred_train.reshape(-1,1),res.reshape(-1,1))))
train_out.columns = ['y_train_true','y_train_score','res']
train_out.to_excel('./PHD2_result/DNN_PHD2_RF113_train_error.xlsx', header=True, index=True)

#测试集数据输出
res=abs(y_test_list-pred_test)
test_out=pd.DataFrame(np.hstack((y_test.reshape(-1,1),pred_test.reshape(-1,1),res.reshape(-1,1))))
test_out.columns = ['y_test_true','y_test_score','res']
test_out.to_excel('./PHD2_result/DNN_PHD2_RF113_test_error.xlsx',header=True,index=True)

print('Test', statistics.loc['Test','R2'])

###################### 实验值与预测值的比较图 ###############################
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.title("DNN",fontsize=25)  # fontweight='bold'
plt.scatter(y_train,pred_train, s=30,marker="^", c="cornflowerblue",
            # label='R2-Train= '+str(round((statistics.loc['Train','R2']),3))
            label='R2-Train= '+str(round(statistics_train[0],3))
            )
plt.scatter(y_test,pred_test, s=30,marker="v", c="darkorange",
            # label='R2-Test= '+str(round((statistics.loc['Test','R2']),3))
            label='R2-Train= '+str(round(statistics_test[0],3))
            )
plt.plot([-2,7.5], [-2,7.5],c='black',lw=2)
plt.xlim(-2,7.5)
plt.ylim(-2,7.5)
plt.xlabel('Experimental Values', fontsize=25)  # fontweight="bold"
plt.ylabel('Predicted Values', fontsize=25)
plt.legend(loc="lower right", fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.savefig("PHD2_result/DNN comparison of values between experimental and test.png",dip = 300)
plt.show()

######################### 残差图 #######################
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.title('DNN',fontsize=25)  # fontweight='bold'
plt.hlines(y=0, xmin=-2, xmax=7.5, color='black', lw=2)
plt.xlim(-2,7.5)
plt.ylim(-2,2)
plt.scatter(pred_train, pred_train-y_train_list, s=30, marker="^", c="cornflowerblue", label='Training Set')
plt.scatter(pred_test, pred_test-y_test_list, s=30, marker="v", c="darkorange", label='Test Set')
plt.xlabel('Predicted values',fontsize=25)
plt.ylabel('Residuals',fontsize=25)
plt.legend(loc="upper right", fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.savefig("PHD2_result/DNN residuals between experimental and test.png", dip = 300)
plt.show()


res_train = y_train_list - pred_train
q_train = math.sqrt(((res-res.mean())**2).sum()/len(res_train))
res_test = y_test_list - pred_test
q_test = math.sqrt(((res_test-res_test.mean())**2).sum()/len(res_test))

plt.figure(figsize=(10,8))
plt.title('DNN',fontsize=25)  # fontweight='bold'
plt.hlines(y=2, xmin=-2, xmax=7.5, color='black', lw=2)
plt.hlines(y=-2, xmin=-2, xmax=7.5, color='black', lw=2)
plt.xlim(-2,7.5)
plt.ylim(-4,4)
plt.scatter(pred_train, (res_train-res_train.mean())/q_train, s=30, marker="^", c="cornflowerblue", label='Training Set')
plt.scatter(pred_test, (res_test-res_test.mean())/q_test, s=30, marker="v", c="darkorange", label='Test Set')
plt.xlabel('Predicted Values',fontsize=25)
plt.ylabel('Standardized Residual',fontsize=25)
plt.legend(loc="upper right", fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.savefig("PHD2_result/Standardized DNN residuals between experimental and test.png", dip = 300)
plt.show()


