import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV #加入网格搜索与交叉验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


def variance_demo():
    """
    删除低方差特征——特征选择
    :return: None
    """
    from sklearn.feature_selection import VarianceThreshold
    data = pd.read_csv("‪C:/Users/amstrong/Desktop/PHD2_fp_1.csv")
    # 1、实例化一个转换器类
    transfer = VarianceThreshold(threshold=1)
    # 2、调用fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])
# variance_demo()

data = pd.read_csv("PHD2_fp_filter_1000_10000.csv")
# print(data.shape)
XX = data.columns[1:-1]
YY = data.columns[-1]
X=data[XX]
Y=data[YY]


# 2)划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=22)

# 3)标准化数据集
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

def knn_model():
# (1) kNN k邻近算法
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=2)
    param = {"n_neighbors": [1,2,3,4,5,6,7]}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5)
    classifier.fit(x_train, y_train)
    return classifier

def SVM_model():
    # (2) SVM 支持向量机
    from sklearn.svm import SVC
    classifier = SVC(probability=True)
    param = {"C": [5],
             'kernel':['rbf']}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5,verbose=2,n_jobs=10)
    classifier.fit(x_train, y_train)
    return classifier

def DT_model():
    # (3) DT 决策树
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=3,criterion='gini')
    param = {"criterion": ['gini'],
             'max_depth':[5],
             'max_features': [None],
             'min_samples_split': [5],
             'min_samples_leaf': [2]}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5,verbose=2,n_jobs=10)
    classifier.fit(x_train, y_train)
    return classifier

def RF_model():
    # (4) RF 随机森林
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    param = {"n_estimators": [30],
             'criterion':['gini'],
             'max_depth':[15],
             'min_samples_split':[7],
             'min_samples_leaf':[1]}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5,verbose=2,n_jobs=10)
    classifier.fit(x_train, y_train)
    return classifier

def LR_model():
    # （5）逻辑回归
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    param = {"solver": ['lbfgs'],
             'C':[3]}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5,verbose=2,n_jobs=10)
    classifier.fit(x_train, y_train)
    return classifier

def ET_model():
    # (6)Extra-Trees（Extremely randomized trees，极端随机树）
    from sklearn.ensemble import ExtraTreesClassifier
    classifier = ExtraTreesClassifier()
    param = {"n_estimators": [80],
             'criterion':['entropy'],
             'min_samples_split': [5],
             'min_samples_leaf': [1],
             'max_depth':[15]}
    classifier = GridSearchCV(classifier, param_grid=param, cv=5,verbose=2,n_jobs=10)
    classifier.fit(x_train, y_train)
    return classifier

def Adabost_model():
    # (7) Adabost
    from sklearn.ensemble import AdaBoostClassifier
    classifier = AdaBoostClassifier()
    param = {"n_estimators": [500],
             'learning_rate':[0.01]}
    classifier = GridSearchCV(classifier,param_grid=param,cv=5, verbose = 2, n_jobs=10)
    classifier.fit(x_train,y_train)
    return classifier

def XGBoost_model():
    # (8) XGBoost
    from xgboost import XGBClassifier as XGBC
    classifier = XGBC(use_label_encoder=False)
    param = {'learn_rate':[0.01],
             "n_estimators": [100],
             "max_deep":[3]}
    classifier = GridSearchCV(classifier,param_grid = param,cv = 5, verbose = 3,n_jobs = 10 )
    classifier.fit(x_train,y_train)
    return classifier

def GBDT_model():
    # (9)GBDT(Gradient Boosting Decision Tree，梯度提升树）
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier()
    param = {'max_features':[0.1],
             'max_depth':[9],
             "n_estimators": [300]}

    classifier = GridSearchCV(classifier,param_grid=param,cv=5,n_jobs=10,verbose=2)
    classifier.fit(x_train,y_train)
    return classifier

# classifier = knn_model()
# classifier = SVM_model()
# classifier = DT_model()
# classifier = RF_model()
# classifier = LR_model()
# classifier = ET_model()
# classifier = Adabost_model()
# classifier = XGBoost_model()
classifier = GBDT_model()

# print("最佳参数：", classifier.best_params_)
# print("交叉验证结果:\n", classifier.cv_results_)
# print("最佳结果：\n", classifier.best_score_)
# print("最佳估计器:\n", classifier.best_estimator_)
# score = classifier.score(x_train, y_train)
# print("训练集的准确率为：\n", score)
# score = classifier.score(x_test, y_test)
# print("测试集的准确率为：", score)

# 5）模型评估

y_train_pred = classifier.predict(x_train) # 训练集预测值
y_test_pred = classifier.predict(x_test) # 测试集预测值
y_test_prob = classifier.predict_proba(x_test)
# classifier.predict
# print(y_test_prob)

def model_evaluation():
    """
    模型评估
    :return:
    """
    from sklearn.metrics import precision_score, recall_score,\
        f1_score, accuracy_score, matthews_corrcoef, \
        cohen_kappa_score, brier_score_loss,auc

    precision = precision_score(y_test,y_test_pred)
    recall = recall_score(y_test,y_test_pred)
    acc = accuracy_score(y_test,y_test_pred)
    mcc = matthews_corrcoef(y_test,y_test_pred)
    f1 = f1_score(y_test,y_test_pred)
    # AUC = auc(x_test,y_test)
    kappa = cohen_kappa_score(y_test,y_test_pred)
    BS = brier_score_loss(y_test,np.array(y_test_prob)[:,1])
    print("Precision:",precision,
          "\nAccuracy:",acc,
          "\nRecall:",recall,
          "\nF1—score:",f1,
          # "\nAUC:",AUC,
          "\nbrier_score:", BS,
          "\nKappa:",kappa,
          "\nMCC:",mcc)

# model_evaluation()


"""混淆矩阵"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 定义画图函数的用来打印误分类矩阵(混淆矩阵)"""
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def test_matrix():
    # 计算测试集的误分类矩阵
    class_names = ['inactive', 'active']
    class_names = np.array(class_names)
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    np.set_printoptions(precision=2)

    #非归一化混淆矩阵
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='GBDT Confusion matrix')
    # plt.savefig("SVM_Confusion_matrix")


    # #归一化混淆矩阵
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Test set Normalized confusion matrix')

# test_matrix()

"""ROC曲线"""
y_score = classifier.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('GBDT ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(fpr, tpr)


###############################################################################################


