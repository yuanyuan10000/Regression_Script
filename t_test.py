from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV #加入网格搜索与交叉验证
import joblib
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef,confusion_matrix,mean_absolute_error,r2_score,mean_squared_error,classification_report

test = pd.read_csv("test_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv")
k_fold = 10  # 10折交叉验证

save_path = "./caco2_result"
model_name = "SVM"
data_name = "caco2"
feature_selection_approach = "RF117"

regression = joblib.load('{}/{}_{}.pkl'.format(save_path, model_name, data_name))
estimator = regression.best_estimator_
print(model_name)

r2_mean_list = []
for i in range(10):
    print("##################### num{} ########################".format(i+1))
    test = shuffle(test)
    x_test = test.iloc[:,1:-1]
    y_test = np.ravel(test.iloc[:,-1])
    per_k = int(x_test.shape[0] / k_fold)  # 每折交叉验证的数据量：140


    r2_list = []
    for k in range(k_fold):
        # 划分数据
        x_valid_k = x_test[k * per_k : (k + 1) * per_k]
        y_valid_k = y_test[k * per_k : (k + 1) * per_k]
        pred_y = regression.predict(x_valid_k)
        r2 = r2_score(y_valid_k, pred_y)
        r2_list.append(r2)
    print(r2_list)
    r2_mean = np.mean(r2_list)
    print(r2_mean)
    r2_mean_list.append(r2_mean)
print("r2_mean_list = ",r2_mean_list)