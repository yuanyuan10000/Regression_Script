
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %matplotlib inline

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import pandas as pd
import molvs as mv

def parent(smiles):
    st = mv.Standardizer() #MolVS standardizer
    try:
        mols = st.charge_parent(Chem.MolFromSmiles(smiles))
        return Chem.MolToSmiles(mols)
    except:
        print ("%s failed conversion"%smiles)
        return "NaN"

# data_new = "E:/WEE1/WEE1_100nm_smiles2.csv"
data = pd.read_csv('./caco2_data/val_MDCK.csv')

data[0] = data.SMILES.apply(parent)
data.to_csv('11.csv',index=False)