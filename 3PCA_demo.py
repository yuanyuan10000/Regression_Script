import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt # 绘图库
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

train = shuffle(pd.read_csv("I:\\1caco2_reg\PHD2_data\\train_random_partition_ECFP6.csv"))
test = shuffle(pd.read_csv("I:\\1caco2_reg\PHD2_data\\test_random_partition_ECFP6.csv"))
x_train = StandardScaler().fit_transform(train.iloc[:,1:-1])
x_test = StandardScaler().fit_transform(test.iloc[:,1:-1])

# 4）PCA降维
transfer = umap.UMAP(n_neighbors=10,
                     min_dist=0.5,
                     random_state=16)
# transfer = TSNE(n_components=2)
# transfer = PCA(n_components=2)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)


# 2.绘制散点图
plt.figure(figsize=(10,8))
plt.scatter(x_train[:,0], x_train[:,1],s=40,marker="^",c="cornflowerblue",label='Training Set')
plt.scatter(x_test[:,0], x_test[:,1], s=40,marker="v",c="darkorange",label='Test Set')

# plt.xlabel('PCA-1',fontsize=25)
# plt.ylabel('PCA-2',fontsize=25)
plt.xlabel('UMAP-1',fontsize=25)
plt.ylabel('UMAP-2',fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("./PHD2_result/dimensionality reduction by UMAP(ECFP6).png",dip=300)
# 3.显示图像
plt.show()

"""
'.'       point marker
','       pixel marker
'o'       circle marker
'v'       triangle_down marker
'^'       triangle_up marker
'<'       triangle_left marker
'>'       triangle_right marker
'1'       tri_down marker
'2'       tri_up marker
'3'       tri_left marker
'4'       tri_right marker
's'       square marker
'p'       pentagon marker
'*'       star marker
'h'       hexagon1 marker
'H'       hexagon2 marker
'+'       plus marker
'x'       x marker
'D'       diamond marker
'd'       thin_diamond marker
'|'       vline marker
'_'       hline marker
"""
