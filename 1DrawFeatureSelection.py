import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#存储 The number of descriptors--kappa 数据
Mutual = pd.read_csv('./PHD2_data/feature_selection_mutual_EGLN1_std.csv')
RF = pd.read_csv('./PHD2_data/feature_selection_RF_EGLN1_std.csv')
F_value = pd.read_csv('./PHD2_data/feature_selection_FValue_EGLN1_std.csv')

# #绘制“The number of descriptors--kappa”图像

plt.figure(figsize=(10,8))
plt.scatter(Mutual.iloc[:,0],Mutual.iloc[:,1],marker='v',linewidths=5)
plt.plot(Mutual.iloc[:,0],Mutual.iloc[:,1],lw = 3,label='Mutual Information')

plt.scatter(Mutual.iloc[:,0],RF.iloc[:,1],marker='^',linewidths=5)
plt.plot(Mutual.iloc[:,0],RF.iloc[:,1],lw = 3,label='Random Forest Importance')

plt.scatter(Mutual.iloc[:,0],F_value.iloc[:,1],marker='d',linewidths=5)
plt.plot(Mutual.iloc[:,0],F_value.iloc[:,1],lw = 3,label='ANOVA F-value')

plt.xlabel("The Number of Descriptors",fontsize=20)
plt.ylabel("Cross Validated R2",fontsize=20)

plt.xticks(np.arange(0,1950,150),fontsize=15)
plt.yticks(np.arange(0.4, 0.9, 0.1),fontsize=15)
plt.grid(True,alpha=0.5)
plt.legend(fontsize=15)

plt.savefig("./PHD2_data/feature selection.png",dip=300)
plt.show()
