# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:56:35 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
from numpy import *
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
from rdkit import DataStructs

train=pd.read_csv('I:\\1caco2_reg\PHD2_data\\train_random_partition.csv')
test=pd.read_csv('I:\\1caco2_reg\PHD2_data\\test_random_partition.csv')

train_smiles=list(train['smiles'])
test_smiles=list(test['smiles'])
fingerprint_list=[]
for smile in train_smiles:
    m = Chem.MolFromSmiles(smile)
    fingerprint_list.append(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024))
    
test['Similarity']=0
    
for index,row in test.iterrows():
    # print(row)
    # print(index+1)
    m=Chem.MolFromSmiles(row['smiles'])
    fp2=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
    all_similarity = []
    similar = []

    for fp1 in fingerprint_list:
        similarity=DataStructs.FingerprintSimilarity(fp1,fp2)
        all_similarity.append(similarity)

    mean_similarity = mean(all_similarity)
    similar.append(mean_similarity)
        # if similarity>max_similar:
        #     max_similar=similarity

    test.loc[index,'Similarity']=similar

test.to_csv('./PHD2_result/PHD2_similarity.csv',index=False)


#####################画相似性分布直方图############################
data = pd.read_csv('PPB_similarity.csv')
y = data[data.columns[-1]]
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.hist(y,bins=30,color = 'teal',edgecolor='#ABB6C8')
plt.grid(axis='y',alpha = 0.05)
plt.xlabel("Similarity to the Training Set",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.savefig("./caco2_result/Similarity of val_caco2 to the Training Set",dip=300)
plt.show()