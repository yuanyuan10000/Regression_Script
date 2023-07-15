import pandas as pd


data = pd.read_csv("./caco2_data/val_caco2_SelectedFeature_RF117_Morgan_moe2d_mol2vec.csv")
print(data.shape)
# XX = data[data.columns[1:-1]]
YY = data[data.columns[-1]]
smiles = data[data.columns[0]]

"找出smiles列表中字符串最长的smile"
smiles_index = 0
len_smile = len(smiles[0])
for index,smile in enumerate(smiles):
    if len(smile) > len_smile:
        len_smile = len(smile)
        smiles_index = index
print("最长smiles长度:",len_smile)  # 256
print("最长smile的索引:",smiles_index) # 353

"找出出smiles长度>150的smiles"
indexs = []
del_smiles = []
for index,smile in enumerate(smiles):
    if len(smile) > 100:
        indexs.append((index+2))
        del_smiles.append(smile)
print(indexs)  # [40, 101, 353, 504, 883]
print(len(del_smiles))



"画smiles长度分布直方图"
len_smiles = []
for smile in smiles:
    len_smiles.append(len(smile))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.hist(len_smiles,label="len_smiles",bins=30,
         # rwidth=0.9,
         color='darkcyan',edgecolor='#ABB6C8')
# plt.plot([150,150],[0,1700],c="r")
plt.xlabel("Length distribution of smiles",fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.grid(axis='y', alpha=0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
# plt.legend()
plt.savefig("./caco2_result/SmiLength(0-150) distribution of data.png",dip=300)
# plt.savefig("./result_PPB/SmiLength distribution of data.png",dip=300)
plt.show()
