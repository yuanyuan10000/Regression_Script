import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("./caco2_result/XGB_caco2_RF117_train_error.csv")
test = pd.read_csv("./caco2_result/XGB_caco2_RF117_test_error.csv")

h_train = pd.read_csv("./caco2_result/h1.csv")
h_test = pd.read_csv("./caco2_result/h2.csv")
h_train = h_train.iloc[:,1:]
h_test = h_test.iloc[:,1:]

leverage_train = [h_train.iloc[i,i] for i in range(h_train.shape[0])]
leverage_test = [h_test.iloc[i,i] for i in range(h_test.shape[0])]

res_train = train.iloc[:,2]
res_test = test.iloc[:,2]


plt.figure(figsize=(10,8))
plt.scatter(res_train, leverage_train, s=10, c="cornflowerblue",label='Training Set')
plt.scatter(res_test, leverage_test, s=10, c="darkorange",label='Test Set')
plt.legend(loc="upper right", fontsize=25)
plt.hlines(y=0.25, xmin=-2, xmax=2, color='silver', lw=2)
plt.vlines(x=0.5943346, ymin=-0, ymax=1, color='silver', lw=2)
plt.vlines(x=-0.5943346, ymin=-0, ymax=1, color='silver', lw=2)
plt.xlim(-1.4,1.4)
plt.ylim(0,0.5)
plt.xlabel('Prediction Error', fontsize=25)  # fontweight="bold"
plt.ylabel('Leverage Value', fontsize=25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
# plt.savefig("./caco2_result/williams_plot.png")
plt.show()


