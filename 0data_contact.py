import pandas as pd

ecfp6 = pd.read_csv('PHD2_data\EGLN3_redup_ECFP6.csv')
ecfp6.drop(columns=['Label'],inplace=True)

mol2vec = pd.read_csv('PHD2_data\EGLN3_redup_mol2vec.csv')
mol2vec.drop(columns=['Unnamed: 0', 'CID', 'Smiles', 'Label'],inplace=True)

rdkit = pd.read_csv('PHD2_data\EGLN3_redup_RDKit_2D.csv')
rdkit.drop(columns=['smiles'],inplace=True)

data = pd.concat([ecfp6,mol2vec,rdkit],axis=1)


data.to_csv('PHD2_data\EGLN3_ECFP6_mol2vec_rdkit.csv',index=False)