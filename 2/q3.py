import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


raw_feature = [] #list
raw_class = [] #list
#raw_test_feature = []
#raw_test_class = []
mean_array = [] #np.array
std_array = []
normalizing_feature = []

def euc_dist(test,train): # all of two are np.array
    temp = test - train
    return(sum(x**2 for x in temp))



with open('seeds.csv', newline='') as seedsfile:
    rows = csv.reader(seedsfile)
    for row in rows:
        raw_feature.append(list(map(float, row[0:7]))) #print(raw_feature)
        raw_class.append(list(map(int, row[7])))       #print(raw_class)
#with open('seeds_test.csv', newline='') as seeds_testfile:
#    rows = csv.reader(seeds_testfile)
#    for row in rows:
#        raw_test_feature.append(list(map(float, row[0:7])))
#        raw_test_class.append(list(map(int, row[7])))



#mean array
mean_array = np.average(np.array(raw_feature), axis=0)
#standard deviation
std_array = np.std(np.array(raw_feature), axis=0)
#normalizing feature
for k in range(np.array(raw_feature).shape[1]):
    normalizing_feature.append( ((np.array(raw_feature)[:,k] - mean_array[k])/std_array[k]).tolist())
normalizing_feature = np.matrix(normalizing_feature).T

#train_feat = [] #np.array
#test_feat = [] #np.array
test_feat = np.append(np.array(normalizing_feature[50:70,:]),np.append(np.array(normalizing_feature[120:140,:]),np.array(normalizing_feature[190:210,:])))
test_feat = test_feat.reshape(int(test_feat.size/7), 7)

train_feat = np.delete(normalizing_feature, [i for i in range(190,210)],0)
train_feat = np.delete(train_feat, [i for i in range(120,139)],0)
train_feat = np.delete(train_feat, [i for i in range(50,69)],0)
print(train_feat)
