import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV #加入网格搜索与交叉验证
import joblib
import joblib
import math
import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef,confusion_matrix,mean_absolute_error,r2_score,mean_squared_error,classification_report
from sklearn.multiclass import OneVsRestClassifier
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

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
    data = np.array([r2, mse, k, kR, r02, r02R, r2, rm2, rm2R])  # 此处删了rm2, rm2R
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
    print("############### KNN ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def SVM_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (2) SVM 支持向量机
    from sklearn.svm import SVR
    reg = SVR()
    print("############### SVR ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def DT_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (3) DT 决策树
    from sklearn.tree import DecisionTreeRegressor
    reg = DecisionTreeRegressor()
    print("############### DT ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def RF_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (4) RF 随机森林
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor()
    print("############### RF ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def ET_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (6)Extra-Trees（Extremely randomized trees，极端随机树）
    from sklearn.ensemble import ExtraTreesRegressor
    reg = ExtraTreesRegressor()
    print("############### ET ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def Adabost_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (7) Adabost
    from sklearn.ensemble import AdaBoostRegressor
    reg = AdaBoostRegressor()
    print("############### Ada ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def XGBoost_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (8) XGBoost
    from xgboost import XGBRegressor as XGBR
    reg = XGBR()
    print("############### XGB ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid.best_estimator_
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

def GBDT_model(param=None, cv=10, verbose=2, n_jobs=10):
    # (9)GBDT(Gradient Boosting Decision Tree，梯度提升树）
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(max_depth=7)
    print("############### GBDT ################")
    if param:
        grid = GridSearchCV(reg, param_grid=param, cv=cv, verbose=verbose, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print(grid.best_params_)
        return grid
    else:
        reg.fit(x_train, np.array(y_train))
        return reg

# # 2)划分数据集
train = pd.read_csv("train_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv")
test = pd.read_csv("test_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv")
x_train = train.iloc[:,1:-1]
y_train = np.ravel(train.iloc[:,-1])
x_test = test.iloc[:,1:-1]
y_test = np.ravel(test.iloc[:,-1])

save_path = "./caco2_result"
model_name = "SVM"
data_name = "caco2"
feature_selection_approach = "RF117"


# 划分数据集1
# data = pd.read_csv("./lmd/we")
# print(data.shape)
# X = data[data.columns[1:-1]]
# Y = data[data.columns[-1]]
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=22)
# 标准化
# std = StandardScaler()
# x_train = std.fit_transform(x_train)
# x_test = std.transform(x_test)



np.random.seed(0)
# regression = knn_model()
regression = SVM_model()
# regression = DT_model()
# regression = RF_model()
# regression = ET_model()
# regression = Adabost_model()
# regression = XGBoost_model()
# regression = GBDT_model()

joblib.dump(regression, '{}/{}_{}.pkl'.format(save_path, model_name, data_name))
# joblib.load('{}/{}_{}.pkl'.format(save_path, model_name, data_name))


# print("最佳参数：", regression.best_params_)
# print("交叉验证结果:\n", regression.cv_results_)
# print("最佳估计器:\n", regression.best_estimator_)
# score = regression.score(x_train, y_train)
# print("训练集的准确率为：\n", score)
# score = regression.score(x_test, y_test)
# print("测试集的准确率为：", score)

# 交叉验证结果
estimator = regression.best_estimator_
y_train_cross_predicted = cross_val_predict(estimator, x_train, y_train, cv=10, n_jobs=14)
Q2 = r2_score(y_train, y_train_cross_predicted)
MSE_CV = mean_squared_error(y_train, y_train_cross_predicted)

# 训练集和测试集的预测值
pred_test = regression.predict(x_test)
pred_train = regression.predict(x_train)

# 计算训练集的统计指标
statistics_train = CalculateMetrics(y_train,pred_train)
statistics_train = np.append(statistics_train,Q2)
statistics_train = np.append(statistics_train,MSE_CV)
# 计算测试集的统计指标
statistics_test = CalculateMetrics(y_test,pred_test)
statistics_test = np.append(statistics_test,Q2)
statistics_test = np.append(statistics_test,MSE_CV)

#输出数据
statistics = pd.DataFrame(np.vstack((statistics_train,statistics_test)))
statistics.columns = ['R2','MSE','k','k\'','r02','r0\'2','r2','rm2','rm\'2','Q2','MSE_CV']
statistics.index = ['Train','Test']
statistics.to_excel('{}/{}_{}_{}_statistics.xlsx'.format(save_path, model_name, data_name, feature_selection_approach),header=True,index=False)

print('Q2_train = {},\nr2_train = {}, \nMSE_train = {}'.format(
    statistics.loc['Train','Q2'], statistics.loc['Train','R2'], statistics.loc['Train','MSE']))
print('Q2_test = {},\nr2_test = {}, \nMSE_test = {}'.format(
    statistics.loc['Test','Q2'], statistics.loc['Test','R2'], statistics.loc['Test','MSE']))


# #测试集误差值输出
res_test = abs(y_test-pred_test)  # 取绝对值
test_out = pd.DataFrame(np.hstack((y_test.reshape(-1,1),pred_test.reshape(-1,1),res_test.reshape(-1,1))))
test_out.columns = ['true_test','pred_test','res']
test_out.to_excel('{}/{}_{}_{}_test_error.xlsx'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=True)
test_out.to_csv('{}/{}_{}_{}_test_error.csv'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=False)
#
# #训练集误差值输出
res_train = abs(y_train-pred_train)
train_out=pd.DataFrame(np.hstack((y_train.reshape(-1,1), pred_train.reshape(-1,1), res_train.reshape(-1,1))))
train_out.columns = ['true_test','pred_test','res']
train_out.to_excel('{}/{}_{}_{}_train_error.xlsx'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=True)
train_out.to_csv('{}/{}_{}_{}_train_error.csv'.format(save_path, model_name,data_name, feature_selection_approach), header=True, index=False)


###################### 实验值与预测值的比较图 ###############################
def ValueComparison_plot():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.title('{}'.format(model_name),fontsize=15)  # fontweight='bold'
    plt.scatter(y_train,pred_train, s=30,marker="^", c="cornflowerblue", label='R2-Train= '+str(round((statistics.loc['Train','R2']),3)))
    plt.scatter(y_test,pred_test, s=30,marker="v", c="darkorange", label='R2-Test= '+str(round((statistics.loc['Test','R2']),3)))
    x = np.arange(-7.5,4.5)
    plt.plot(x,x,c='black',lw=3)
    plt.xlabel('Experimental Values', fontsize=15)
    plt.ylabel('Prediction Values', fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    # plt.savefig("{}/{} comparison of values between experimental and test.png".format(save_path, model_name),dip = 300)
    plt.show()
    return None

######################### 残差图 #######################
def residual_plot():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.title('{}'.format(model_name),fontsize=15)  # fontweight='bold'
    plt.scatter(pred_train, pred_train-y_train, s=30, marker="^", c="cornflowerblue", label='train')
    plt.scatter(pred_test, pred_test-y_test, s=30, marker="v", c="darkorange", label='test')
    plt.hlines(y=0, xmin=-3, xmax=0, color='black', lw=3)
    plt.xlabel('Predicted values',fontsize=15,fontweight='bold')
    plt.ylabel('Residuals',fontsize=15,fontweight='bold')
    plt.legend(loc="upper right", fontsize=15,fontweight='bold')
    plt.xticks(fontsize = 15,fontweight='bold')
    plt.yticks(fontsize = 15,fontweight='bold')
    plt.tight_layout()
    # plt.savefig(".{}/{} residuals between experimental and test.png".format(save_path, model_name), dip = 300)
    plt.show()
    return None

