from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np


data = pd.read_csv('PHD2_data\EGLN1_redup.csv')

#################### Murcko Scaffold #######################
# 获取Murcko骨架：GetScaffoldForMol()
smi_gen = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi) for smi in data["smiles"]]
print("共有原始分子{}个".format(data.shape[0]))
smi_gen_1 = list(set(smi_gen))
print("共生成不重复的Murcko骨架{}个".format(len(smi_gen_1)))
frequency_scaff = []
for index in range(len(smi_gen_1)):
    c = smi_gen.count(smi_gen_1[index])
    frequency_scaff.append(c)
print(len(frequency_scaff))

# 另存为csv文件
# c = np.stack((smi_gen_1,frequency_scaff))
# result = pd.DataFrame(c.T)
# result.to_csv("22.csv")

#################### Generic Framework #######################
# 产生generic骨架：MakeScaffoldGeneric()
mol_core = []
for smi in data['smiles']:
    smile = Chem.MolFromSmiles(smi)   # 将分子转换为mol格式
    core = MurckoScaffold.GetScaffoldForMol(smile)  # 生成Murcko骨架（mol）
    mol_core.append(core)
gen_mcore = list(map(MurckoScaffold.MakeScaffoldGeneric, mol_core))   # 生成generic骨架(mol格式)
gen_mcore = [Chem.MolToSmiles(mol) for mol in gen_mcore]    # 将mol转换为smile
print("共有原始分子{}个".format(len(gen_mcore)))
gen_mcore_1 = list(set(gen_mcore))    # 去重
print("共生成generic骨架{}个".format(len(gen_mcore_1)))

# # 计算骨架出现频率
frequency_scaff = []
for index in range(len(gen_mcore_1)):
    c = gen_mcore.count(gen_mcore_1[index])
    frequency_scaff.append(c)
print(len(frequency_scaff))

# 另存为csv文件
c = np.stack((gen_mcore_1,frequency_scaff))
result = pd.DataFrame(c.T,  index=None, columns= ["Smiles", "Frequency"])
result.to_csv("generic_scaffold_analysis_PHD2.csv")


################# 可视化分子 ######################

data = pd.read_csv("generic_scaffold_analysis_PHD2.csv")
data.sort_values(by='Frequency',ascending=False,inplace=True)
smiles = data["Smiles"][:10]
mols = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)

# 多个分子按照grid显示
img = Draw.MolsToGridImage(mols,
                           subImgSize=(500,500),
                           # legends=['' for x in mols],
                           molsPerRow=5)     # 一行几个分子

## 单个分子转图片
# opts = DrawingOptions()
# m = Chem.MolFromSmiles('OC1C2C1CC2')
# opts.includeAtomNumbers=True
# opts.bondLineWidth=2.8
# draw = Draw.MolToImage(m, options=opts)
# draw.save('/Users/zeoy/st/drug_development/st_rdcit/img/mol10.jpg')

img.save('./PHD2_result/GenericFramework_top10_PHD2.jpg')
img.show()

"""
note:
len([mol for mol in mols if mol is not None]) 查看有效分子个数
mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]  将smi转化为sdf
core = [MurckoScaffold.GetScaffoldForMol(m) for m in mols]
len([mol for mol in core if mol is not None])
Draw.MolsToGridImage(mols[:482])
Draw.MolToImage(Chem.MolFromSmiles("[11CH3]N(C)c1ccc(cc1)C2=CC(=O)c3ccc(O)c(O)c3O2")).show()
"""

