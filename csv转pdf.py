from rdkit import Chem
import pandas as pd
readpath='E:/WEE1/wee1_rd50_40ek.csv'
savepath='E:/WEE1/wee1_rd50_40ek.sdf'
data=pd.read_csv(readpath)
#cid=data['CID'].values
smiles=data['smiles'].values

def save_sdf(filename):
    mols=[Chem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
    mols=[mol for mol in mols if mol is not None]
   # for i,m in enumerate(mols):
   #     m.SetProp("CID",str(cid[i]))

    w = Chem.SDWriter(filename)
    for m in mols: w.write(m)
    w.close()
save_sdf("E:/WEE1/wee1_rd50_40ek.sdf")
