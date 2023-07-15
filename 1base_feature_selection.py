from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

Variancethreshold = 0
Corrthreshold=0.9

np.random.seed(0)

D = pd.read_csv('PHD2_data\EGLN1_ECFP6_mol2vec_rdkit.csv')
data = D.iloc[:, 1:-1]
smiles = D['smiles']
label = D['Label']
# print(data.shape)

# 获取变量名
feature_name = data.columns.values

# 低方差滤波
sel = VarianceThreshold(threshold=Variancethreshold)
data_deleted = sel.fit_transform(data)

# 保存变量的选择情况
selsupport=pd.DataFrame(sel.get_support(),dtype=int)
feature_name = feature_name.T[sel.get_support()].T

#计算删除了多少变量
before_VT_row,before_VT_col = data.shape   # (1753, 2540)
after_VT_row,after_VT_col = data_deleted.shape   # (1753, 2487)
print("VarianceThreshold删去"+str(before_VT_col-after_VT_col)+"个特征变量")

#高相关滤波
# np.corrcoef()  计算矩阵的相关系数，返回Pearson乘积矩相关系数的矩阵
corrcoef = np.corrcoef(data_deleted.T)
r,c = corrcoef.shape   # (2487, 2487)
transform = np.array([True]*r)
deletelist = []
for i in range(r):
    for k in range(i+1,r):
        if corrcoef[i][k] > Corrthreshold:
            if i not in deletelist and k not in deletelist:
                deletelist.append(k)
                transform[k]=False


#计算删除了多少变量
print("相关系数高的删去"+str(len(deletelist))+"个特征变量")


# #保存变量的选择情况
corrsupport = pd.DataFrame(transform,dtype=int)
data_deleted_corr = data_deleted.T[transform].T
feature_name = feature_name.T[transform].T

print(feature_name)
print("剩余特征数:"+str(len(feature_name)))


data = pd.DataFrame(data_deleted_corr)
data.columns=list(feature_name)

egln1 = pd.concat([smiles,data,label],axis=1)
egln1.to_csv('PHD2_data/feature_selection_EGLN1.csv',index=False)
selected_col = egln1.columns
egln2 = pd.read_csv('PHD2_data\EGLN2_ECFP6_mol2vec_rdkit.csv',usecols=selected_col)
egln3 = pd.read_csv('PHD2_data\EGLN3_ECFP6_mol2vec_rdkit.csv',usecols=selected_col)
# external = pd.read_csv("PHD2_data/external_EGLN1_ECFP6_mol2vec_rdkit.csv", usecols=selected_col)
external = pd.read_csv("PHD2_data/pose_filter_lig_redup_mol2vec_ECFP6_RDkit2D.csv", usecols=selected_col)

# 标准化
scl = StandardScaler()

x_egln1 = egln1.iloc[:,1:-1]
x_egln2 = egln2.iloc[:,1:-1]
x_egln3 = egln3.iloc[:,1:-1]
x_external = external.iloc[:,1:-1]

std_egln1 = pd.DataFrame(scl.fit_transform(x_egln1),columns=feature_name)
std_egln2 = pd.DataFrame(scl.transform(x_egln2),columns=feature_name)
std_egln3 = pd.DataFrame(scl.transform(x_egln3),columns=feature_name)
std_external = pd.DataFrame(scl.transform(x_external),columns=feature_name)


D_egln1 = pd.concat([egln1['smiles'],std_egln1,egln1['Label']],axis=1)
D_egln2 = pd.concat([egln2['smiles'],std_egln2,egln2['Label']],axis=1)
D_egln3 = pd.concat([egln3['smiles'],std_egln3,egln3['Label']],axis=1)
D_external = pd.concat([external['smiles'],std_external,external['Label']],axis=1)

D_egln1.to_csv('PHD2_data/feature_selection_EGLN1_std.csv',index=False)
D_egln2.to_csv('PHD2_data/feature_selection_EGLN2_std.csv',index=False)
D_egln3.to_csv('PHD2_data/feature_selection_EGLN3_std.csv',index=False)
D_external.to_csv('PHD2_data/feature_selection_pose_filter_lig_redup_std.csv',index=False)



