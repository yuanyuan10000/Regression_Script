import rdkit
import pandas as pd
from rdkit import Chem

# # 找出无效分子
# smiles = pd.read_csv('./caco2_data/raw_caco2_MaxLen150.csv')
# invalid = []
# for i,smi in enumerate(smiles['SMILES']):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         invalid.append(i)
# print(invalid)
#--------------------------------------------------------------------------
import rdkit
import pandas as pd
from rdkit import Chem

smiles = pd.read_csv('./PHD2_data/data_PHD2.csv')

prop = smiles['Label']
v_mol = []
v_prop = []
for i in range(smiles.shape[0]):
    smi = smiles['SMILES'][i]
    # print(smi)
    mol = Chem.MolFromSmiles(smi)
    # print(mol)
    if mol is not None:
        # v_prop.append(prop[i])
        # mol.SetProp('Label',prop[i])
        v_mol.append(mol)

writer = Chem.SDWriter('./PHD2_data/data_PHD2.sdf')
for mol in v_mol:
    writer.write(mol)
writer.close()
