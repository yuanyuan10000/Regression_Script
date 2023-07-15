import pandas as pd
import random
from rdkit import Chem

augmentation = 10
max_len = 200
data = pd.read_csv(r"C:\Users\201\Desktop\raw_caco2_argumention.csv")
smiles_data = data['Smiles']
label_data = data['Label']

def randomSmiles(mol):
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)

def smile_augmentation(smile, augmentation, max_len):
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = randomSmiles(mol)
        if len(smiles) <= max_len :
            s.add(smiles)
            if len(s) == augmentation:
                break

    return list(s)

all_alternative_smi = []
else_smi_index = []
for index,smi in enumerate(smiles_data):
    alternative_smi = smile_augmentation(smi, augmentation, max_len)
    if len(alternative_smi) != 3:
        else_smi_index.append(index)
    all_alternative_smi.extend(alternative_smi)



print(len(all_alternative_smi))
print('生成随机smiles长度不为argumention的索引列表为：',else_smi_index)

# mol = Chem.MolFromSmiles('c1cc(NC(CCCCCNC(CS)=O)=O)c2ncccc2c1')
# s = set()
# for i in range(1000):
#     smiles = randomSmiles(mol)
#     if len(smiles) <= max_len and smiles not in s:
#         s.add(smiles)
#         if len(s) == augmentation:
#             break


