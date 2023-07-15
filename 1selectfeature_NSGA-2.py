# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:09:15 2020

@author: Administrator
"""

# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari



#Importing required modules
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import os
from sklearn.model_selection import cross_val_predict
import random

os.chdir(r'E:\akt predict\new predict')

#First function to optimize
def function1(solution):
    value = [-solution[i].sum() for i in range(solution.shape[0])]
    return value


#Second function to optimize
def function2(solution, x_train, y_train, x_test, y_test):
    r2_list = []
    for curr_solution in solution:
        selected_elements_indices = np.where(curr_solution == 1)[0]##输出满足条件 (即非0) 元素的坐标 
        x_train_selected = x_train[:, selected_elements_indices]#输出被选择的描述符的值
        #x_test_selected=x_test[:, selected_elements_indices]
        
        clf = RandomForestRegressor(random_state=0,n_jobs=12)
        try:
            y_train_pred = cross_val_predict(clf,x_train_selected,y_train,cv=5,n_jobs=14)
        except:
            r2_list.append(0)
            continue
        #SV_classifier.fit(x_train_selected, y_train)
        #y_test_pred = SV_classifier.predict(x_test_selected)
        r2=r2_score(y_train, y_train_pred)
        #r2=r2_score(y_test, y_test_pred)
        r2_list.append(r2)
    return r2_list



#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(parent1,parent2,max_feature):
    cross_point = np.random.randint(0,max_feature)
    #交换父代100个特征，生成两个子代
    offspring1=parent1[:max(cross_point-50,0)]+parent2[max(cross_point-50,0):min(cross_point+50,max_feature)]+parent1[min(cross_point+50,max_feature):]
    offspring2=parent2[:max(cross_point-50,0)]+parent1[max(cross_point-50,0):min(cross_point+50,max_feature)]+parent2[min(cross_point+50,max_feature):]
    offsprings=mutation(np.array([offspring1,offspring2]),max_feature)
    return offsprings

#Function to carry out the mutation operator
def mutation(offspring_crossover,max_feature,num_mutations=2):    
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        mutation_prob = random.random()
        if mutation_prob >0.5:
            mutation_idx = np.random.randint(low=0, high=max_feature, size=num_mutations)
            offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover


def dominated_ratio(solution,new_solution):
    merge=np.vstack((solution,new_solution))
    function1_values = function1(merge)
    function2_values = function2(merge,x_train, y_train, x_test, y_test)
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])#得到每一级的非支配曲面
    num_solution_dominate_new_solution=0
    num_new_solution_dominate_solution=0
    index_solution=list(range(len(solution)))
    index_new_solution=list(range(len(solution),len(solution)+len(new_solution)))
    priority_plato=[]
    for plato in non_dominated_sorted_solution:
        priority_plato+=plato
        num_solution_dominate_new_solution+=len([index for index in index_new_solution if index not in priority_plato])
        num_new_solution_dominate_solution+=len([index for index in index_solution if index not in priority_plato])
    print('num_new_solution_dominate_solution',num_new_solution_dominate_solution)
    print('num_solution_dominate_new_solution',num_solution_dominate_new_solution)
    try:
        ratio=num_new_solution_dominate_solution/(num_new_solution_dominate_solution+num_solution_dominate_new_solution)
    except:
        ratio=0.5
    return ratio

    
#Main program starts here#########################
dataset='AKT1-IC50-single-protein'


#读取训练数据和验证集
x_train=pd.read_excel('./shuffle data/'+dataset+' training set_feature_clean.xlsx')
y_train=pd.read_excel('./shuffle data/'+dataset+' training set_label.xlsx',header=None)
y_train=np.ravel(y_train.iloc[:,0])

x_test=pd.read_excel('./shuffle data/'+dataset+' test set_feature_clean.xlsx')
y_test=pd.read_excel('./shuffle data/'+dataset+' test set_label.xlsx',header=None)
y_test=np.ravel(y_test.iloc[:,0])

feature_name=x_train.columns.values.tolist()
x_train=np.array(x_train)
x_test=np.array(x_test)


#Initialization
pop_size = 50
max_feature=len(feature_name)
max_gen = 100#最大迭代次数


solution = np.zeros(shape=(pop_size,max_feature))
for i in range(solution.shape[0]):
    for k in range(500):
        number=random.randint(0,solution.shape[1]-1)#前闭后闭区间
        solution[i,number]=1
        prob = random.random()
        if prob > 0.4:
            solution[i,number]=0
            
    print(solution[i].sum())        
#solution=np.array(pd.read_excel(r'./result/'+dataset+' feature select-NSGA-2.xlsx'))
#print(solution.shape)

solution=solution.astype(np.int32)
print(solution)

ratio_list=[]
for gen_no in range(1,max_gen+1):
    print(gen_no)
        
    solution2 = solution[:]
    #Generating offsprings,并且子代和父代合并
    while(len(solution2)<=2*pop_size):
        a1 = random.randint(0,pop_size-1)#生成指定范围的随机数
        b1 = random.randint(0,pop_size-1)
        offsprints=crossover(list(solution[a1]),list(solution[b1]),max_feature)
        solution2=np.append(solution2,offsprints,axis=0)#对随机的两个染色体（个体）进行杂交
        
    function1_values2 = function1(solution2)
    function2_values2 = function2(solution2,x_train, y_train, x_test, y_test)
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])#得到每一级的非支配曲面
    #计算拥挤度（如果排序等级一样，优选选择拥挤度大的个体，拥挤度越大，适应度也越大）
    #拥挤度的维度和非支配曲面是一样的    
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))

    new_solution= []#new_solution是index表示的个体
    for i in range(0,len(non_dominated_sorted_solution2)):
        #得到合并集的第i个非支配曲面的第j个元素的index的集合
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] )
                                            for j in range(0,len(non_dominated_sorted_solution2[i]))]

        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])#对每一个非支配曲面的个体进行拥挤度排序，并输出index
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]#对每一个非支配曲面的个体进行拥挤度排序，并输出代表个体的数字
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    new_solution = np.array([solution2[i] for i in new_solution])#new_solution变成了二进制染色体
    ratio=dominated_ratio(solution,new_solution)
    print('generation',gen_no,'ratio',ratio)
    ratio_list.append(ratio)
    #更新父代
    solution=new_solution
    #输出solution
    out=pd.DataFrame(solution)
    out.columns=feature_name
    out.to_excel(r'./result/'+dataset+' feature select-NSGA-2.xlsx',index=False)

    
x_axis=list(range(1,max_gen+1))
##PLOT
plt.figure(figsize=(10,8))
plt.grid(True)
plt.plot(x_axis,ratio_list,'bo')
plt.plot(x_axis,ratio_list,'b')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Generation',fontsize=20)
plt.ylabel('Dominance Ratio',fontsize=20)
#plt.title('Training and validation acc',fontsize=25)
plt.legend(fontsize=18)
plt.savefig('./result/'+dataset+' feature select-NSGA-2.png',dpi=200)
plt.show()


###输出最终解的最高非支配曲面######
function1_values = function1(solution)
function2_values = function2(solution,x_train, y_train, x_test, y_test)
non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])#得到每一级的非支配曲面

index_list=np.array([])
cols = np.array([])
r2_result = np.array([])

for index in non_dominated_sorted_solution[0]:
    print(index)
    selected_elements_indices = np.where(solution[index] == 1)[0]##输出满足条件 (即非0) 元素的坐标 
    x_train_selected = x_train[:, selected_elements_indices]#输出被选择的描述符的值
    x_test_selected=x_test[:, selected_elements_indices]
    
    SV_classifier = SVR(gamma='scale')
    y_train_pred = cross_val_predict(SV_classifier,x_train_selected,y_train,cv=5,n_jobs=12)
    print('最优解r2为',r2_score(y_train, y_train_pred),'最优解描述符数量为',solution[index].sum())
    
    #存储每一次的r2和变量个数
    index_list=np.append(index_list,index)
    cols = np.append(cols,solution[index].sum())
    r2_result = np.append(r2_result,r2_score(y_train, y_train_pred))

print(non_dominated_sorted_solution)
#存储 The number of descriptors--r2 数据

df = np.vstack((index_list,cols,r2_result))
NumberOfDescriptors = pd.DataFrame(df)
NumberOfDescriptors = NumberOfDescriptors.T
NumberOfDescriptors.to_excel('./result/'+dataset+' feature select-NSGA-2-statistic.xlsx',header=False,index=False)



'''
###################比较最优解的重复性###
dataset='AKT1-IC50-single-protein'

solution=pd.read_excel(r'./result/'+dataset+' feature select-NSGA-2.xlsx')
solution=np.array(solution)
index_list=[6,40]
dic_best_solution={}
for index in index_list:
    selected_elements_indices = np.where(solution[index] == 1)[0]
    print(selected_elements_indices)
'''


'''
######比较三个最优解的公约数##############
dataset='AKT1-IC50-single-protein'

solution=pd.read_excel(r'./result/'+dataset+' feature select-NSGA-2.xlsx')

feature_name=solution.columns.values.tolist()
solution=np.array(solution)
solution1 = list(np.where(solution[6] == 1)[0])
solution2 = list(np.where(solution[11] == 1)[0])
solution3 = list(np.where(solution[9] == 1)[0])

common_feature=[]
final_solution = list(set(solution1).intersection(solution2,solution3))
for i in final_solution:
    common_feature.append(feature_name[i])

common_feature=pd.DataFrame(np.array(common_feature))
common_feature.to_excel(r'./result/common feature-NSGA-2.xlsx',header=False,index=False)

#求出最优解的特征###
best_feature=[]
for i in list(solution1):
    best_feature.append(feature_name[i])
best_feature=pd.DataFrame(np.array(best_feature))
best_feature.to_excel(r'./result/best feature-NSGA-2.xlsx',header=False,index=False)
'''



#######################选择最优解后挑选数据集并且输出#######################
dataset='AKT1-IC50-single-protein'
index=6

solution=pd.read_excel(r'./result/'+dataset+' feature select-NSGA-2.xlsx')
feature_name=solution.columns.values.tolist()

solution=np.array(solution)

#读取训练数据和验证集
x_train=pd.read_excel('./shuffle data/'+dataset+' training set_feature_clean.xlsx')
y_train=pd.read_excel('./shuffle data/'+dataset+' training set_label.xlsx',header=None)
y_train=np.ravel(y_train.iloc[:,0])

x_test=pd.read_excel('./shuffle data/'+dataset+' test set_feature_clean.xlsx')
y_test=pd.read_excel('./shuffle data/'+dataset+' test set_label.xlsx',header=None)
y_test=np.ravel(y_test.iloc[:,0])


x_train=np.array(x_train)
x_test=np.array(x_test)

selected_elements_indices = np.where(solution[index] == 1)[0]
x_train_selected = x_train[:, selected_elements_indices]#输出被选择的描述符的值
x_test_selected=x_test[:, selected_elements_indices]

x_train_selected=pd.DataFrame(x_train_selected)
x_train_selected.to_excel('./shuffle data/'+dataset+' training set_NSGA-2_selected.xlsx',header=None,index=None)

x_test_selected=pd.DataFrame(x_test_selected)
x_test_selected.to_excel('./shuffle data/'+dataset+' test set_NSGA-2_selected.xlsx',header=None,index=None)


    