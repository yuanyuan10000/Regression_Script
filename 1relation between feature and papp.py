import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



cols = ['NumAliphaticHeterocycles','PEOE_VSA11','mol2vec-001','bit339','mol2vec-097',
        'mol2vec-076','PEOE_VSA13','mol2vec-054','mol2vec-235','mol2vec-150','Label']

data = pd.read_csv("I:\\1caco2_reg\PHD2_data\EGLN1_ECFP6_mol2vec_rdkit.csv",usecols=cols)



s_size = 17
f_size = 20


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25,8))

axes[0,0].scatter(data['NumAliphaticHeterocycles'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[0,0].set_xlabel('NumAliphaticHeterocycles',fontsize=f_size)
axes[0,0].set_ylabel('Label',fontsize=f_size)

axes[0,1].scatter(data['PEOE_VSA11'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[0,1].set_xlabel('PEOE_VSA11',fontsize=f_size)
axes[0,1].set_ylabel('Label',fontsize=f_size)

axes[0,2].scatter(data['mol2vec-001'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[0,2].set_xlabel('mol2vec-001',fontsize=f_size)
axes[0,2].set_ylabel('Label',fontsize=f_size)

axes[0,3].scatter(data['bit339'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[0,3].set_xlabel('bit339',fontsize=f_size)
axes[0,3].set_ylabel('Label',fontsize=f_size)

axes[0,4].scatter(data['mol2vec-097'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[0,4].set_xlabel('mol2vec-097',fontsize=f_size)
axes[0,4].set_ylabel('Label',fontsize=f_size)

axes[1,0].scatter(data['mol2vec-076'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[1,0].set_xlabel('mol2vec-076',fontsize=f_size)
axes[1,0].set_ylabel('Label',fontsize=f_size)

axes[1,1].scatter(data['PEOE_VSA13'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[1,1].set_xlabel('PEOE_VSA13',fontsize=f_size)
axes[1,1].set_ylabel('Label',fontsize=f_size)

axes[1,2].scatter(data['mol2vec-054'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[1,2].set_xlabel('mol2vec-054',fontsize=f_size)
axes[1,2].set_ylabel('Label',fontsize=f_size)

axes[1,3].scatter(data['mol2vec-235'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[1,3].set_xlabel('mol2vec-235',fontsize=f_size)
axes[1,3].set_ylabel('Label',fontsize=f_size)

axes[1,4].scatter(data['mol2vec-150'], data['Label'], s = s_size,color = 'cornflowerblue')
axes[1,4].set_xlabel('mol2vec-150',fontsize=f_size)
axes[1,4].set_ylabel('Label',fontsize=f_size)

plt.savefig('./PHD2_result/1relation between feature and pIC50_2')

plt.tight_layout()
plt.show()
