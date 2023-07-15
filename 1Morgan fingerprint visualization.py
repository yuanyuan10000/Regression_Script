import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem,Draw

mol = Chem.MolFromSmiles('CCN1C(=O)COc2nc(c3ccc(cc3)[C@@]4(N)C[C@@](C)(O)C4)c(cc12)c5ccccc5')
mol = Chem.AddHs(mol)###加氢
AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
AllChem.MMFFOptimizeMolecule(mol)


bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=2048, bitInfo=bi)
print(bi)

mfp2_svg = Draw.DrawMorganBit(mol, 378, bi, useSVG=True)#(bit-name-1)
f = open("test.html",'w')#保存成html然后用浏览器打开
f.write(mfp2_svg)
f.close()

print(mfp2_svg)


