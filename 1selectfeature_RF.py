import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import time
start = time.process_time()
from xgboost import  XGBRegressor

np.random.seed(0)

from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score,mean_squared_error,roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.model_selection import cross_val_predict

# 读取训练数据和验证集
data = pd.read_csv('I:\\1caco2_reg\PHD2_data\\feature_selection_EGLN1_std.csv')
X_train = data.iloc[:, 1:-1]
y_train = data.iloc[:,-1]

feature_name = X_train.columns.values.tolist()
X_train=np.array(X_train)


# 输出特征重要性排序
estimator = RandomForestRegressor(random_state=5, n_jobs=12)
estimator.fit(X_train, y_train)

importance = pd.DataFrame(estimator.feature_importances_)
feature_name = pd.DataFrame(feature_name)
importance = pd.concat([feature_name, importance], axis=1)
importance.columns = ['feature_name', 'score']
importance.to_excel('I:\\1caco2_reg\PHD2_data\\feature_importance_RF.xlsx',index=False)



#########类似互信息的方法一样，选择n个特征，然后制作特征数和模型结果的图###
importance = pd.read_excel('I:\\1caco2_reg\PHD2_data\\feature_importance_RF.xlsx')
importance = importance.sort_values(by="score", ascending=False)

# 获得特征的index排序
feature_index = importance.index.values.tolist()

# 用来存储特征数和f1
cols = np.array([])
r2_list = np.array([])
# for n in range(10, X_train.shape[1], 100):
for n in range(100,150, 1):
    selected_elements_indices = feature_index[:n]
    X_train_selected = X_train[:, selected_elements_indices]

    y_train_pred = cross_val_predict(GradientBoostingRegressor(), X_train_selected, y_train, cv=10, n_jobs=32)
    r2 = r2_score(y_train, y_train_pred)
    print('当特征值的个数为{n}时，r2为{r2}'.format(n=X_train_selected.shape[1], r2=r2))

    # 存储每一次的kappa和变量个数
    cols = np.append(cols, n)
    r2_list = np.append(r2_list, r2)

# 绘制“The number of descriptors--kappa”图像
plt.figure(figsize=(10, 8))
plt.grid(True)
plt.scatter(cols, r2_list, c='b', marker='^', linewidths=5, edgecolors='b')
plt.plot(cols, r2_list, '-b', lw=3)
plt.xlabel("The Number of Descriptors(Selected by Random Forest)", fontsize=20)
plt.ylabel("Cross Validated R2", fontsize=20)
plt.ylim(0, 1)
# plt.xticks(range(10, importance.shape[0], 200), fontsize=20)
plt.xticks(range(100,150, 10), fontsize=20)
plt.yticks(np.arange(0, 1.01, 0.1), fontsize=20)
# plt.savefig('./result/' + dataset + ' feature select-RF.png', dpi=200)
plt.show()

# 存储 The number of descriptors--kappa 数据
df = np.vstack((cols, r2_list))
NumberOfDescriptors = pd.DataFrame(df)
NumberOfDescriptors = NumberOfDescriptors.T
# NumberOfDescriptors.to_csv('./PHD2_data/feature_selection_RF_EGLN1_std.csv',index=False)
NumberOfDescriptors.to_csv('./PHD2_data/fineTuning_feature_selection_RF_EGLN1_std.csv',index=False)



##################当选择好n时，输出选择的数据集###########
import numpy as np
import pandas as pd


importance = pd.read_excel('I:\\1caco2_reg\PHD2_result\\1feature_importance_RF.xlsx')
importance = importance.sort_values(by="score", ascending=False)
n=113

selected_feature = list(importance['feature_name'])[:n]
selected_feature.insert(0,"smiles")
selected_feature.insert(len(selected_feature), 'Label')


egln1 = pd.read_csv('I:\\1caco2_reg\PHD2_data\\feature_selection_EGLN1_std.csv',usecols=selected_feature)
egln2 = pd.read_csv('I:\\1caco2_reg\PHD2_data\\feature_selection_EGLN2_std.csv',usecols=selected_feature)
egln3 = pd.read_csv('I:\\1caco2_reg\PHD2_data\\feature_selection_EGLN3_std.csv',usecols=selected_feature)
external = pd.read_csv('I:\\1caco2_reg\PHD2_data/feature_selection_pose_filter_lig_redup_std.csv',usecols=selected_feature)


egln1.to_csv('./PHD2_data/RF113_feature_selection_EGLN1_std.csv',index=False)
egln2.to_csv('./PHD2_data/RF113_feature_selection_EGLN2_std.csv',index=False)
egln3.to_csv('./PHD2_data/RF113_feature_selection_EGLN3_std.csv',index=False)
external.to_csv('./PHD2_data/RF113_feature_selection_pose_filter_std.csv',index=False)