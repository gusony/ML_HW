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
def comp_knm(par_index,i,j):
    knm = hypP[par_index][0]*np.exp((-1)*hypP[par_index][1]*((train_data_x[i]-train_data_x[j])**2)/2)+hypP[par_index][2]+hypP[par_index][3]*train_data_x[i]*train_data_x[j]
    if i ==j :
        knm+=1
    return(knm)

def get_C(hpi):# hyper parameter index
    C = []
    for i in range(len(train_data_x)):
        temp = []
        for j in range(len(train_data_x)):
            temp.append(comp_knm(hpi,i,j))
        C.append(temp)
    return(np.matrix(C))

def mean_list(par_index,test_data_index):
    k_list = []
    for i in range(60):
        k_list.append(comp_knm(par_index, i, test_data_index))

    return (np.matrix(k_list).T.dot(pinv(C_matrix_list[par_index])).dot(np.matrix(train_data_t)))

def var_matrix(par_index,test_data_index):
    c = comp_knm(par_index, test_data_index,test_data_index)

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


#bluild four C matrix

for i in range(4):
    C_matrix_list.append(get_C(i))



f,part2 = plt.subplots(1,1)
#part2.plot(train_data_x,np.random.multivariate_normal(zero_list, C_matrix_list[0]).flatten(),'-')
plt.scatter(one_list,np.random.multivariate_normal(zero_list, C_matrix_list[0]).flatten())
plt.show()


