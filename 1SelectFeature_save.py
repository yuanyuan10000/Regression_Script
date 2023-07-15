import matplotlib
from dask.optimization import inline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据预处理：特征工程和标准化
data = pd.read_csv("./data/filter_base_caco2_morgan_mol2vec_moe2d.csv")
X = data.iloc[:, 1:-1]
Y = data.iloc[:,-1]

std = StandardScaler()
# std = MinMaxScaler()
x = pd.DataFrame(std.fit_transform(X))
x.to_csv("filter_base_caco2_morgan_mol2vec_moe2d_StandScacle.csv")


'''
#  F检验(标准化后)
data = pd.read_csv("./data/filter_base_caco2_morgan_mol2vec_moe2d.csv")
x = data.iloc[:, 1:-1]
y = data.iloc[:,-1]
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor as RF
F, pvalues_f = f_classif(x,y)

k = F.shape[0] - (pvalues_f > 0.05).sum()
X_fsF = SelectKBest(f_classif, k=k).fit_transform(x, y)

# a = pd.DataFrame(X_fsF)
# a.to_csv('caco2_final_descreptions.csv')

score = cross_val_score(RF(n_estimators=10,random_state=0),X_fsF,y,cv=5).mean()
print(score)

# %matplotlib inline
import matplotlib.pyplot as plt
score = []
for i in range(10,2400,1000):
    X_fschi = SelectKBest(f_classif, k=i).fit_transform(x, y)
    once = cross_val_score(RF(n_estimators=10,random_state=0),X_fschi,y,cv=5).mean()
    score.append(once)
plt.ylabel("Cross Validated R2")
plt.grid(True)
plt.plot(range(10,2400,100),score)
plt.show()'''