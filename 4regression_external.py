import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV #加入网格搜索与交叉验证
import joblib
import math
import itertools
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef,confusion_matrix,mean_absolute_error,r2_score,mean_squared_error,classification_report
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
    data = np.array([R2, mse, k, kR, r02, r02R, r2, rm2, rm2R])  # 此处删了rm2, rm2R
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



external = pd.read_csv('./PHD2_data/RF113_feature_selection_pose_filter_std.csv')
x_external = external.iloc[:,1:-1]
y_external = np.ravel(external.iloc[:,-1])


save_path = "./PHD2_result"
model_name = "SVM"
data_name = "PHD2"
feature_selection_approach = "RF113"

estimator = joblib.load('{}/{}_{}.pkl'.format(save_path, model_name, data_name))

pred_test = estimator.predict(x_external)

statistics_test = CalculateMetrics(y_external, pred_test)
print("test R2: {}\nPHD2 MSE: {}".format(statistics_test[0],statistics_test[1]))

# pred_caco2 = estimator.predict(x_caco2)
# statistics_caco2 = CalculateMetrics(y_caco2, pred_caco2)
# print("val_caco2 R2: {}\nval_caco2 MSE: {}".format(statistics_caco2[0],statistics_caco2[1]))

pred_MDCK = estimator.predict(x_MDCK)
statistics_MDCK = CalculateMetrics(y_MDCK, pred_MDCK)
print("val_MDCK R2: {}\nval_MDCK MSE: {}".format(statistics_MDCK[0],statistics_MDCK[1]))

# statistics = pd.DataFrame(np.vstack((statistics_caco2,statistics_MDCK)))
# statistics.columns = ['R2','MSE','k','k\'','r02','r0\'2','r2','rm2','rm\'2']
# statistics.index = [caco2, MDCK]
# statistics.to_csv('{}/{}_{}_{}_statistics.xlsx'.format(save_path, model_name, data_name, feature_selection_approach))

