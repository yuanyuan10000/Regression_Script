import pandas as pd
import numpy as np

data = pd.read_csv("PHD2_data\Cluster_RF113_feature_selection_EGLN1_std.csv")
lengths = [0, 25, 50, 75, 100, 125, 150]
train_idx = []
for i in range((len(lengths)-1)):
    idx = data[(data['smiles'].str.len() >= lengths[i]) & (
            data['smiles'].str.len() < lengths[i + 1])].sample(frac=0.8).index
    train_idx.extend(idx)
## 划分训练集
train_data = data[data.index.isin(train_idx)]
test_data = data[~data.index.isin(train_idx)]
train_data.to_csv("train_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv",index=False)
test_data.to_csv("test_SelectedFeature_RF117_caco2_MaxLen150_morgan_mol2vec_moe2d.csv",index=False)

# 划分测试集和验证集
# test_idx = []
# for i in range((len(lengths)-1)):
#     idx = data[(data['SMILES'].str.len() >= lengths[i]) & (
#             data['SMILES'].str.len() < lengths[i + 1])].sample(frac=1.0).index
#     test_idx.extend(idx)
#
# test_data = data[data.index.isin(test_idx)]
# # val_data = data[~data.index.isin(test_idx)]

test_data.to_csv("test_SelectedFeature_RF104_PPB_morgan_mol2vec_moe2d.csv")
# val_data.to_csv("val_filter_RF152_caco2_morgan_mol2vec_moe2d.csv")