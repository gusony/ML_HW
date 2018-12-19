# ML HW3 problem1
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


train_data_x = []
train_data_t = []
test_data_x = []
test_data_t = []
C_matrix_list = []
hypP = [[1,4,0,0],
        [0,0,0,1],
        [1,4,0,5],
        [1,64,10,0]] # const
zero_list = [0 for i  in range(60)]
one_list = [1 for i in range(60)]


#input : hyper parameter index
def comp_knm(par_index,xn,xm):
    return(hypP[par_index][0]*np.exp((-1)*hypP[par_index][1]*((xn-xm)**2)/2)+hypP[par_index][2]+hypP[par_index][3] * xn * xm)

def get_C(hpi):# hyper parameter index
    C = []
    for i in range(len(train_data_x)):
        temp = []
        for j in range(len(train_data_x)):
            if i == j :
                result = 1
            result += comp_knm(hpi,train_data_x[i],train_data_x[j])
            temp.append(result)
        C.append(temp)
    return(np.matrix(C))

def get_k_list(par_index, test_data_index):
    k_list = []
    for i in range(60):
        k_list.append(comp_knm(par_index, train_data_x[i], test_data_x[test_data_index]))
    return (k_list)

def mean_of_x(par_index,test_data_index, k_list):
    return (np.matrix(k_list).dot(pinv(C_matrix_list[par_index]).dot(np.matrix(train_data_t).T))  )

def var_matrix(par_index,test_data_index, k_list):
    c = comp_knm(par_index, test_data_t[test_data_index],test_data_t[test_data_index])
    print(c,np.matrix(k_list).dot(pinv(C_matrix_list[par_index])).dot(np.matrix(k_list).T))
    return(c - np.matrix(k_list).dot(pinv(C_matrix_list[par_index])).dot(np.matrix(k_list).T))

#read data
with open('gp.csv', newline='') as rowfile:
    rows = csv.reader(rowfile)
    i = 0
    for row in rows:
        if i < 60:
            train_data_x.append(float(row[0]))
            train_data_t.append(float(row[1]))
        else:
            test_data_x.append(float(row[0]))
            test_data_t.append(float(row[1]))
        i+=1

f,flt = plt.subplots(1,4)
#f,flt2 = plt.subplots(4,1)

#bluild four C matrix
for HyperParameterIndex in range(4):
    C_matrix_list.append(get_C(HyperParameterIndex))
    mean_t_of_x = []
    cvar_t_of_x = []
    for test_index in range(60):
        k_list = get_k_list(HyperParameterIndex, test_index)
        mean_t_of_x.append(mean_of_x(HyperParameterIndex, test_index, k_list).flatten().tolist()[0])
        #cvar_t_of_x.append(var_matrix(HyperParameterIndex, test_index, k_list).flatten().tolist()[0])
    flt[HyperParameterIndex].plot(test_data_x, mean_t_of_x)
    #flt2[HyperParameterIndex].scatter(test_data_x, cvar_t_of_x)
plt.show()


