import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# 读取训练数据和验证集
data = pd.read_csv('I:\\1caco2_reg\PHD2_data\\feature_selection_EGLN1_std.csv')
X_train = np.array(data.iloc[:, 1:-1])
y_train = data.iloc[:,-1]



# 用来存储特征数和f1
cols = np.array([])
r2_list = np.array([])

for n in range(10, X_train.shape[1], 100):
    # 按照一定比例选取特征
    X_train_selected = SelectKBest(f_regression, k=n).fit_transform(X_train, y_train)
    # 得到列数，即特征数目
    r, c = X_train_selected.shape

    y_train_pred = cross_val_predict(GradientBoostingRegressor(),
                                     X_train_selected, y_train, cv=10, n_jobs=14)
    r2 = r2_score(y_train, y_train_pred)
    print('当特征值的个数为{n}时，r2为{r2}'.format(n=X_train_selected.shape[1],r2=r2))
    # 存储每一次的r2和变量个数
    cols = np.append(cols,n)
    r2_list = np.append(r2_list,r2)

# 绘制“The number of descriptors--kappa”图像
plt.figure()
plt.grid(True, alpha = 0.3)
plt.scatter(cols, r2_list)
plt.plot(cols, r2_list)
plt.xlabel("The Number of Descriptors(Selected by ANOVA F_value)")
plt.ylabel("Cross Validated R2")
plt.ylim(0, 1)
# plt.xticks(range(10, X_train.shape[1], 200), fontsize=20)
plt.yticks(np.arange(0, 1.01, 0.1))
# plt.savefig('./result/' + dataset + ' feature select-F_value.png', dpi=200)
plt.show()

# 存储 The number of descriptors--kappa 数据
df = np.vstack((cols, r2_list))
NumberOfDescriptors = pd.DataFrame(df)
NumberOfDescriptors = NumberOfDescriptors.T
NumberOfDescriptors.to_csv('./PHD2_data/feature_selection_FValue_EGLN1_std.csv',index=False)