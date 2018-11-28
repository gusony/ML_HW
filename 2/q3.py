import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math

###################################################################################
raw_feature = [] #list
raw_class = [] #list
mean_array = [] #np.array
std_array = []
normalizing_feature = []
test_class = []
train_class = []
###################################################################################
def all_test_data_euc_dist(test,train): #return (60,150)
    result = []
    for i in range(test.shape[0]):
        temp = []
        for j in range(train.shape[0]):
            temp.append(np.linalg.norm(test[i]-train[j]))
        result.append(temp)
    return(np.array(result))

def get_one_data_class_result(K, arr):
    result = np.array([0,0,0])
    for i in range(K):
        result.itemset( train_class[arr.tolist().index(min(arr))]-1, result[train_class[arr.tolist().index(min(arr))]-1]+1 )
        arr = np.delete(arr, arr.tolist().index(min(arr)))
    return(result.tolist().index(max(result))+1)

def get_all_class_result(K,dis_array): # (60,150)
    result = []
    for n in range(dis_array.shape[0]):
        result.append(get_one_data_class_result(K, dis_array[n,:]))
    return(np.array(result))

def get_one_data_class_by_V(V,dis_arr):
    result = np.array([0,0,0])
    #dis_arr.shape[0] :150
    for n in range(dis_arr.shape[0]):
        if dis_arr[n] <= V:
            result.itemset( train_class[n]-1 ,result[train_class[n]-1]+1)
    return(result.tolist().index(max(result))+1)

def get_all_class_by_V(V,dis_array):
    result = []
    for n in range(dis_array.shape[0]):
        result.append(get_one_data_class_by_V(V, dis_array[n,:]))
    return(np.array(result)) #(60,)

def compute_acc_rate(result):
    acc = 0
    for i in range(result.shape[0]):
        if (round(result[i]) == round(test_class[i])): #對答案 test_class (60,)
            acc +=1
    return(acc/result.shape[0])
###################################################################################
with open('seeds.csv', newline='') as seedsfile:
    rows = csv.reader(seedsfile)
    for row in rows:
        raw_feature.append(list(map(float, row[0:7]))) #print(raw_feature)
        raw_class.append(list(map(int, row[7]))[0])       #print(raw_class)

#part1
#step2
#mean array
mean_array = np.average(np.array(raw_feature), axis=0)

#standard deviation
std_array = np.std(np.array(raw_feature), axis=0)

#normalizing feature
for k in range(np.array(raw_feature).shape[1]):
    normalizing_feature.append( ((np.array(raw_feature)[:,k] - mean_array[k])/std_array[k]).tolist())
normalizing_feature = np.matrix(normalizing_feature).T

#divide raw feature data into two sets, test & train
test_feat = np.append(np.array(normalizing_feature[50:70,:]),np.append(np.array(normalizing_feature[120:140,:]),np.array(normalizing_feature[190:210,:])))
test_feat = test_feat.reshape(int(test_feat.size/7), 7)
train_feat = np.delete(normalizing_feature, [i for i in range(190,210)],0)
train_feat = np.delete(train_feat, [i for i in range(120,140)],0)
train_feat = np.delete(train_feat, [i for i in range(50,70)],0)

test_class = np.append(np.array(raw_class[50:70]), np.append( np.array(raw_class[120:140]),np.array(raw_class[190:210])))
train_class = np.delete(raw_class, [i for i in range(190,210)],0)
train_class = np.delete(train_class, [i for i in range(120,140)],0)
train_class = np.delete(train_class, [i for i in range(50,70)],0)
#step3 get euclidean distance
dis_array = all_test_data_euc_dist(test_feat, np.array(train_feat))

#step4
x_K =[]
accu_rate_by_K = []
class_result = []
for K in range(1,11):
    print('K:',K)
    x_K.append(K)
    accu_rate_by_K.append(compute_acc_rate(get_all_class_result(K, dis_array)))

f,part1 = plt.subplots()
part1.set_title('HW2, Q3, Part1')
part1.set_xlabel('K')
part1.set_ylabel('accuracy rate')
part1.plot(x_K,accu_rate_by_K)





#part2 step4
x_V = []
accu_rate_by_V = []
for V in range(2,11):
    print('V:',V)
    x_V.append(V)
    accu_rate_by_V.append(compute_acc_rate(get_all_class_by_V(V, dis_array)))

f,part2 = plt.subplots()
part2.set_title('HW2, Q3, Part2')
part2.set_xlabel('V')
part2.set_ylabel('accuracy rate')
part2.plot(x_V,accu_rate_by_V)


plt.show()





