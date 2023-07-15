import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV #加入网格搜索与交叉验证
import joblib
import math
import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef,confusion_matrix,mean_absolute_error,r2_score,mean_squared_error,classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle


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


def knn_model(param=None, cv=10, verbose=2, n_jobs=10):
# (1) kNN k邻近算法
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " KNN " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def SVM_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (2) SVM 支持向量机
    from sklearn.svm import SVR
    reg = SVR()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " SVM " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def DT_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (3) DT 决策树
    from sklearn.tree import DecisionTreeRegressor
    reg = DecisionTreeRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " DT " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def RF_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (4) RF 随机森林
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " RF " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def ET_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (6)Extra-Trees（Extremely randomized trees，极端随机树）
    from sklearn.ensemble import ExtraTreesRegressor
    reg = ExtraTreesRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " ET " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def Adaboost_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (7) Adabost
    from sklearn.ensemble import AdaBoostRegressor
    reg = AdaBoostRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-" * 35 + " Ada " + "-" * 35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def XGBoost_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (8) XGBoost
    from xgboost import XGBRegressor as XGBR
    reg = XGBR()
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-" * 35 + " XGB " + "-" * 35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def GBDT_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (9)GBDT(Gradient Boosting Decision Tree，梯度提升树）
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(max_depth=7)
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-"*35 + " GBDT " + "-"*35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def lightGBM_model(param=None,verbose=2,n_jobs=10):
    # LightGBM:梯度提升框架
    from lightgbm import LGBMRegressor
    reg = LGBMRegressor()
    if param:
        grid = GridSearchCV(reg, param_grid=param,verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        print("-" * 35 + " LightGBM " + "-" * 35)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg



data = pd.read_csv('I:\\1caco2_reg\PHD2_data\RF113_feature_selection_EGLN1_std.csv')
X = data.iloc[:,1:-1]
# X = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train,y_test = np.array(y_train),np.array(y_test)




# train = shuffle(pd.read_csv("PHD2_data/train_Cluster_RF113_feature_selection_EGLN1_std.csv"),random_state=22)
# test = shuffle(pd.read_csv("PHD2_data/test_Cluster_RF113_feature_selection_EGLN1_std.csv"),random_state=22)

# x_train = train.iloc[:,1:-1]
# y_train = np.ravel(train.iloc[:,-3])
# x_test = test.iloc[:,1:-1]
# y_test = np.ravel(test.iloc[:,-3])

save_path = "./PHD2_result/"
model_name = "XGBoost"
data_name = "PHD2"
feature_selection_approach = "RF113"


np.random.seed(0)
# regression = knn_model(param={"n_neighbors": list(range(1,11,1))},n_jobs=50)   # 3
# regression = SVM_model(param={'C':list(range(1,20,1)),
#                               'epsilon':np.arange(0.01,0.31,0.05),  # 0.06
#                               'gamma':['scale','auto',0.1,0.2,0.3,0.4]},n_jobs=50)
# regression = DT_model(param={'max_depth':list(range(1,76,5))},n_jobs=50)
# regression = RF_model(param={"n_estimators": list(range(10,510,50)),   #150
#                              'max_depth':list(range(1,76,10))}, n_jobs=50) #20

# regression = ET_model()
# regression = Adaboost_model(param={"learning_rate": [0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.3],
#                                   'n_estimators':list(range(50,850,50))},n_jobs=50)
regression = XGBoost_model(param = {"n_estimators": list(range(10,860,50)),   #160
                                  'max_depth': list(range(3,8,1)),   #6
                                  "learning_rate": [0.01,0.03,0.05,0.08,0.1,0.2,0.3],}, n_jobs=10)  # 0.1
# regression = GBDT_model(param={'learning_rate':[0.01,0.03,0.05,0.08,0.1,0.2,0.3],
#                                'n_estimators':list(range(50,850,50)),
#                                'max_depth':list(range(3,8,1))}, n_jobs=10)
# regression = lightGBM_model(param={"learning_rate": [0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.3],
#                                    'max_depth':list(range(3,8,1)),
#                                    'num_leaves':[4,8,16,32,64,128,256]})

## 输出模型
joblib.dump(regression, '{}/{}_{}.pkl'.format(save_path, model_name, data_name))

###################################################################################################
# 加载模型
regression = joblib.load('{}/{}_{}.pkl'.format(save_path, model_name, data_name))
pred_test = regression.predict(x_test)
pred_train = regression.predict(x_train)

estimator = regression.best_estimator_


y_train_cross_predicted = cross_val_predict(estimator, x_train, y_train, cv=10, n_jobs=14)
Q2 = r2_score(y_train, y_train_cross_predicted)
MSE_CV = mean_squared_error(y_train, y_train_cross_predicted)
# print(Q2, MSE_CV)


statistics_train = CalculateMetrics(y_train,pred_train)
statistics_train = np.append(statistics_train,Q2)
statistics_train = np.append(statistics_train,MSE_CV)

# 计算测试集的统计指标
statistics_test = CalculateMetrics(y_test,pred_test)
statistics_test = np.append(statistics_test,Q2)
statistics_test = np.append(statistics_test,MSE_CV)

# 输出测试集的统计指标
statistics = pd.DataFrame(np.vstack((statistics_train,statistics_test)))   # 横向合并np.vstack()
statistics.columns = ['R2','MSE','k','k\'','r02','r0\'2','r2','rm2','rm\'2','Q2','MSE_CV']
statistics.index = ['Train','Test']
statistics.to_excel('{}/{}_{}_{}_statistics.xlsx'.format(save_path, model_name, data_name, feature_selection_approach),header=True)

# 打印统计指标
print('Q2_train = {},\nr2_train = {}, \nMSE_train = {}'.format(
    statistics.loc['Train','Q2'], statistics.loc['Train','R2'], statistics.loc['Train','MSE']))
print('Q2_test = {},\nr2_test = {}, \nMSE_test = {}'.format(
    statistics.loc['Test','Q2'], statistics.loc['Test','R2'], statistics.loc['Test','MSE']))

# #测试集误差值输出
res_test = y_test-pred_test  # 取绝对值
test_out = pd.DataFrame(np.hstack((y_test.reshape(-1,1),pred_test.reshape(-1,1),res_test.reshape(-1,1))))   # 纵向合并np.hstack()
test_out.columns = ['true_test','pred_test','res']
# test_out.to_excel('{}/{}_{}_{}_test_error.xlsx'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=True)
test_out.to_csv('{}/{}_{}_{}_test_error.csv'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=False)

# #训练集误差值输出
res_train = y_train - pred_train
train_out=pd.DataFrame(np.hstack((y_train.reshape(-1,1), pred_train.reshape(-1,1), res_train.reshape(-1,1))))
train_out.columns = ['true_test','pred_test','res']
# train_out.to_excel('{}/{}_{}_{}_train_error.xlsx'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=True)
train_out.to_csv('{}/{}_{}_{}_train_error.csv'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=False)





###################### 实验值与预测值的比较图 ###############################

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.title('{}'.format(model_name),fontsize=25)  # fontweight='bold'
plt.scatter(y_train,pred_train, s=30,marker="^", c="cornflowerblue",
            # label='R2-Train= '+str(round(r2_score(y_train, pred_train), 3))
            label='R2-Train= '+str(round(statistics_train[0],3))
            )

plt.scatter(y_test,pred_test, s=30,marker="v", c="darkorange",
            # label='R2-Test= '+str(round(r2_score(y_test, pred_test),3))
            label='R2-Test= '+str(round(statistics_test[0],3))
            )
plt.plot([-2,7.5], [-2,7.5],c='black',lw=2)
plt.xlim(-2,7.5)
plt.ylim(-2,7.5)
plt.xlabel('Experimental Values', fontsize=25)  # fontweight="bold"
plt.ylabel('Predicted Values', fontsize=25)
plt.legend(loc="lower right", fontsize=25)
plt.xticks(list(range(-2,8)),fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.savefig("{}/{} comparison of values between experimental and test.png".format(save_path, model_name))
plt.show()


res_train = y_train - pred_train
q_train = math.sqrt(((res_train-res_train.mean())**2).sum()/len(res_train))
res_test = y_test - pred_test
q_test = math.sqrt(((res_test-res_test.mean())**2).sum()/len(res_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.title('{}'.format(model_name),fontsize=25)  # fontweight='bold'
plt.hlines(y=2, xmin=-2, xmax=7.5, color='black', lw=2)
plt.hlines(y=-2, xmin=-2, xmax=7.5, color='black', lw=2)
plt.xlim(-2,7.5)
plt.ylim(-6,6)
plt.scatter(pred_train, (res_train-res_train.mean())/q_train, s=30, marker="^", c="cornflowerblue", label='Training Set')
plt.scatter(pred_test, (res_test-res_test.mean())/q_test, s=30, marker="v", c="darkorange", label='Test Set')
plt.xlabel('Predicted Values',fontsize=25)
plt.ylabel('Standardized Residual',fontsize=25)
plt.legend(loc="upper right", fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.savefig("{}/{} Standardized residuals between experimental and test.png".format(save_path, model_name), dip = 300)
plt.show()

