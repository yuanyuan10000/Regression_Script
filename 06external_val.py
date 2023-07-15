import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


##################当选择好n时，输出选择的数据集###########


data = pd.read_csv('./PHD2_data/RF113_feature_selection_EGLN1_std.csv')
selected_col = data.columns
selected_morgan_col = data.columns[1:12].tolist()
selected_mol2vec_col = data.columns[12:89].tolist()
selected_rdkit_col = data.columns[89:-1].tolist()

# len(selected_morgan_col) + len(selected_mol2vec_col) + len(selected_rdkit_col)

smiles = pd.read_csv('./11/fp/lig_redup_ECFP6.csv', usecols=['smiles'])
selected_morgan = pd.read_csv('./11/fp/lig_redup_ECFP6.csv', usecols=selected_morgan_col)
selected_rdkit = pd.read_csv('./11/fp/lig_redup_RDKit_2D.csv', usecols=selected_rdkit_col)
selected_mol2vec = pd.read_csv('./11/fp/lig_redup_mol2vec.csv',usecols=selected_mol2vec_col)

val = pd.concat([smiles, selected_morgan, selected_mol2vec, selected_rdkit],axis=1)
val = val[val['mol2vec-284'].notnull()]

D = pd.read_csv('PHD2_data/feature_selection_EGLN1.csv', usecols=selected_col)

scl = StandardScaler()
x_train = scl.fit_transform(D.iloc[:,1:-1])
x_val = scl.transform(val.iloc[:,1:])

lig = pd.concat([smiles, pd.DataFrame(x_val)], axis = 1)
lig.to_csv('RF113_feature_selection_molcular_std.csv')

################# 模型的验证 #####################
import joblib

regression = joblib.load('PHD2_result\SVM_PHD2.pkl')


pred_val = regression.predict(x_val)
pred_val = pd.DataFrame(pred_val)

smi = val['smiles'].tolist()

result = pd.concat([pd.DataFrame(smi), pred_val],axis=1)
result.columns = ['smiles','pred_value']
result.to_csv('193w_SVM_predicted.csv',index=False)

result.sort_values(by='pred_value', ascending=True, inplace=True)


active_mol = result.iloc[:round(result.shape[0]/5), :]   # 0.564492-2.986138
active_mol.to_csv('386400_SVM_predicted_active 0.564492-2.986138.csv', index=False)

active_smi = active_mol['smiles']
active_smi.to_csv('386400_SVM_predicted_active 0.564492-2.986138.smi', index=False,header=False)








# import math
# import pandas as pd
# from tqdm import tqdm
#
# data = pd.read_csv('./11/fp/lig_redup_RDKit_2D.csv', usecols=['smiles'])
# data.columns = ['Smiles']
#
# data.insert(loc=1, column='Label', value=None)
# data.insert(loc=0, column='CID', value=None)
# data.to_csv('./11/lig_redup.txt',sep='\t',index=False)
# data.to_csv('./11/lig_redup.csv',index=False)
#
#
#
# ########################################################
# file_name = './11/lig_redup.csv'
#
# def cut_df(file_name, n):
#     df = pd.read_csv(file_name)
#     df_num = len(df)
#     every_epoch_num = math.floor((df_num/n))
#     for index in tqdm(range(n)):
#         file_name1 = f'./11/lig_redup_{index}.csv' # 切割后的文件名
#         file_name2 = f'./11/lig_redup_{index}.smi'  # 切割后的文件名
#         if index < n-1:
#             df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
#         else:
#             df_tem = df[every_epoch_num * index:]
#         df_tem.to_csv(file_name1, index=False)
#         df_tem.to_csv(file_name2, sep='\t', index=False)
#
#
# cut_df(file_name, 5)



####################################################################################
# 拼接数据

from molvs import standardize_smiles

smi = pd.read_csv('./11/pair/smi_final.csv')
raw_mol = pd.read_csv('./11/pair/193w_SVM_predicted.csv')

  
raw_smi_std = [standardize_smiles(i) for i in raw_mol['smiles'].values]
raw_mol['smiles'] = raw_smi_std

data = pd.merge(smi, raw_mol, how='left')

data.to_csv('smi_final_active.csv')