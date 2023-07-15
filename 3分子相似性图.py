import pandas as pd
import numpy as np
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem,Draw
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import codecs

def write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
target='PHD2'
def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return fp

df_train = pd.read_csv('I:\\1caco2_reg\PHD2_data\\train_random_partition.csv')
# df_train = df_train[df_train['label'] == 1]

df_test =  pd.read_csv('I:\\1caco2_reg\PHD2_data\\train_random_partition.csv')
# df_test = df_test[df_test['label'] == 1]
# cid=df_test['CID'].values

smi_train=np.squeeze(df_train['smiles'].values)
train_fp=[morgan_fp(s) for s in smi_train]

smi_test=np.squeeze(df_test['smiles'].values)
test_fp=[morgan_fp(s) for s in smi_test]


collect_nbrs=[]
for i in range(len(test_fp)):
    sim_ref = DataStructs.BulkTanimotoSimilarity(test_fp[i],train_fp)
    nbrs = sorted(sim_ref,reverse=True)
    collect_nbrs.append(nbrs)
#    collect_nbrs.append(sim_ref)

# 画热图
collect_nbrs=np.array(collect_nbrs)
f, ax = plt.subplots(figsize = (16,8))
# plt.title('Similarity Matrix of SYK',fontsize=15,fontweight='bold')
sns.heatmap(collect_nbrs,ax=ax,vmax=1.0, vmin=0,cmap='Pastel1')
# Greys Purples Greens YlGnBu Pastel2
plt.xlabel('Molecules in Training Set',fontsize=20)   # ,fontweight='bold'
plt.ylabel('Molecules in Test Set',fontsize=20)   # ,fontweight='bold'
plt.tick_params(labelsize=13)
plt.savefig('./PHD2_result/heatmap.png')
plt.show()