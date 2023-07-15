import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


"""数值分布直方图"""
# train = pd.read_csv("train_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv")
# test = pd.read_csv("test_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv")
# # x_train=train.iloc[:,1:-1]
# y_train=np.ravel(train.iloc[:,-1])
# # x_test=test.iloc[:,1:-1]
# y_test=np.ravel(test.iloc[:,-1])

data = pd.read_csv('I:\\1caco2_reg\PHD2_data\RF113_feature_selection_EGLN1_std.csv')
X = data.iloc[:,1:-1]
# X = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train,y_test = np.array(y_train),np.array(y_test)


import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))

## 画多个变量图
plt.hist(y_train,bins=30,label="train",density=False,color = 'cornflowerblue',edgecolor='#ABB6C8')
plt.hist(y_test,bins=30,label="test",density=False,color = 'darkorange',edgecolor='#ABB6C8')
plt.grid(axis='y',alpha = 0.3)
plt.xlabel("Distribution of Values between Training Set and Test Set",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=17)
plt.tight_layout()
plt.savefig("./PHD2_result/value distribution between train and test",dpi = 300)
plt.show()



#### 画单个图
data = pd.read_csv("I:\\1caco2_reg\PHD2_data\EGLN1_redup.csv")
print(data.shape)
X = data[data.columns[0]]
Y = data[data.columns[-1]]
# Y = data[data.columns[2]]

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(10,8))
# sns.distplot(Y,bins=30,color = 'brown',hist=True, kde=False,
#              # kde_kws={"color": "k", "lw": 3, "label": "KDE"},
#              # hist_kws={ "linewidth": 3,"alpha": 1, "color": "brown"}
#              )
plt.hist(Y,bins=30,density=True,color = 'brown',edgecolor='#ABB6C8')
plt.grid(axis='y',alpha = 0.3)
plt.xlabel("logIC50",fontsize=17)
# plt.xlabel("fup_log10",fontsize=25)
plt.ylabel("Density",fontsize=17)
# plt.ylabel("Frequency",fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig("./PHD2_result/value distribution of PHD",dpi = 300)

plt.show()

