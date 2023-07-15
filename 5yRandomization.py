# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:53:24 2017

@author: zjn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM, SVR
from sklearn.feature_selection import SelectKBest
from xgboost import XGBRegressor as XGBR

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR

# np.random.seed(0)
# train = pd.read_csv("./data/train_raw_caco2_morgan_mol2vec_moe2d.csv")
# test = pd.read_csv("./data/test_raw_caco2_morgan_mol2vec_moe2d.csv")
# X_train = train.iloc[:,1:-1]
# y_train=np.ravel(train.iloc[:,-1])
# X_test=test.iloc[:,1:-1]
# y_test=np.ravel(test.iloc[:,-1])
#
# clf = KNeighborsRegressor()
#
# n=10 #迭代次数
# result=np.zeros(n)#存储迭代n次的结果
#
# for i in range(1,n+1):
#     np.random.shuffle(y_train)
#     CV_r2 = cross_val_score(clf,X_train,y_train,cv=10,scoring='r2')
#     result[i-1] = CV_r2.mean();
#     print(i)
#
# NumberOfDescriptors = pd.DataFrame(result)
# NumberOfDescriptors = NumberOfDescriptors.T
# NumberOfDescriptors.to_excel("result.xlsx",header=False,index=False)
#
#
# sns.set_style("whitegrid")
# sns.distplot(result,bins=9,color='c')
# plt.plot([0.71,0.71],[-0.9,9],'salmon')
# plt.plot([-1,1],[0,0],'salmon')
# plt.xlabel("Q2")
# plt.ylabel("Density")
# plt.ylim(-0.5,9)
# plt.xlim(-1,1)
# # plt.savefig("output/yRandomization_3.tif",dpi=500)
# plt.show()

np.random.seed(0)

train = pd.read_csv("I:\\1caco2_reg\PHD2_data\\train_random_partition.csv")
test = pd.read_csv("I:\\1caco2_reg\PHD2_data\\test_random_partition.csv")
X_train = train.iloc[:,1:-1]
y_train=np.ravel(train.iloc[:,-1])
X_test=test.iloc[:,1:-1]
y_test=np.ravel(test.iloc[:,-1])

clf = SVR()

n = 100  # 迭代次数
result = np.zeros(n)  # 存储迭代n次的结果

for i in range(1, n + 1):
    np.random.shuffle(y_train)
    y_train_cross_predicted = cross_val_predict(clf, X_train, y_train, cv=10, n_jobs=14)
    Q2 = r2_score(y_train, y_train_cross_predicted)
    result[i - 1] = Q2
    print(i)

result = pd.DataFrame(result)
result.to_excel("./PHD2_result/y_random.xlsx", header=False, index=False)

# sns.set_style("whitegrid")
# sns.distplot(result, bins=9, color='c')
# plt.plot([0.71, 0.71], [-0.9, 9], 'salmon')
# plt.plot([-1, 1], [0, 0], 'salmon')
# plt.xlabel("Q2", 'Times New Roman')
# plt.ylabel("Density", 'Times New Roman')
# plt.ylim(-0.5, 13)
# plt.xlim(-1, 1)
# # plt.savefig("result/y_random.jpg", dpi=500)
# plt.show()



### 作图
sns.set_style("whitegrid")
sns.distplot(result, bins=15, color='darkcyan')
plt.plot([0.768,0.768],[-0.75,40],'salmon')
plt.plot([-1,1],[0,0],'salmon')
plt.xlabel("Q2",fontsize=17)
plt.ylabel("Density",fontsize=17)
plt.ylim(-0.5,35)
plt.xlim(-1,1)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig("./PHD2_result/yRandomization.tif",dpi=500)
plt.show()