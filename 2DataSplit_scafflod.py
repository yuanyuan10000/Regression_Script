import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

data = pd.read_csv('I:\\1caco2_reg\PHD2_data\RF113_feature_selection_EGLN1_std.csv')
smi_list = data['smiles'].to_list()
mol_list = [Chem.MolFromSmiles(s) for s in smi_list]
# smi_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) for mol in mol_list]
# mol_scaffolds = [Chem.MolFromSmiles(smi_scaffold) for smi_scaffold in smi_scaffolds]




# 基于Murcko骨架聚类
Murcko_scaffolds = {}
MurckClusters_list = []

Generic_scaffolds = {}
GenericClusters_list = []

idx_Murcko = 1
idx_Generic = 1

for mol in mol_list:
    Murcko_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    Generic_smi = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(Murcko_smi)))


    if Murcko_smi not in Murcko_scaffolds.keys():
        Murcko_scaffolds[Murcko_smi] = idx_Murcko
        idx_Murcko += 1
    cluster_id = Murcko_scaffolds[Murcko_smi]
    MurckClusters_list.append(cluster_id)

    if Generic_smi not in Generic_scaffolds.keys():
        Generic_scaffolds[Generic_smi] = idx_Generic
        idx_Generic += 1
    cluster_id_Generic = Generic_scaffolds[Generic_smi]
    GenericClusters_list.append(cluster_id_Generic)

print("Num of dataset:",len(mol_list))
print("Num of Murcko scaffolds in dataset:",len(Murcko_scaffolds.keys()))
print("Num of Generic scaffolds in dataset:",len(Generic_scaffolds.keys()))

data['Murck_Num'] = MurckClusters_list
data['Generic Num'] = GenericClusters_list

data.to_csv('PHD2_data\Cluster_RF113_feature_selection_EGLN1_std.csv',index=False)
###################################################################################################

data = shuffle(pd.read_csv('PHD2_data\Cluster_RF113_feature_selection_EGLN1_std.csv'),random_state=55)
data = data.reset_index().drop(columns=['index'])

unsplit_idx = []
train_idx = []

for i in range(1,len(set(data['Generic Num']))+1):
    if (data['Generic Num']==i).sum() >=5:
        idx = data[data['Generic Num']==i].sample(frac=0.8).index
        train_idx.extend(idx)
    else:
        idx = data[data['Generic Num'] == i].index
        unsplit_idx.extend(idx)

idx = data.loc[unsplit_idx].sample(frac=0.8).index
train_idx.extend(idx)


train = shuffle(data[data.index.isin(train_idx)])
test = shuffle(data[~data.index.isin(train_idx)])


train.to_csv('PHD2_data/train_Cluster_RF113_feature_selection_EGLN1_std.csv',index=False)
test.to_csv('PHD2_data/test_Cluster_RF113_feature_selection_EGLN1_std.csv',index=False)





