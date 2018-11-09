# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math

origin_x = []
origin_t = []
N_list = [10]#,15,30,80]
M = 7
s = 0.1

def sigmoid_func(x):
    return(1/(1+math.exp(x)))

def SN(phi_matrix):
    result_sn = phi_matrix.T * phi_matrix
    result_sn = result_sn + 10**-6*np.eye(result_sn.shape[0])
    return(result_sn)


def mN(phi_matrix,SN):
    return(SN.dot(phi_matrix.T * np.matrix(origin_t)))


#read data
with open('1_data.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        origin_x.append(float(row[0]))
        origin_t.append(float(row[1]))

for N in N_list:
    phi_j_of_x_matrix = np.matrix([])
    SN_matrix = np.matrix([])
    mN_matrix = np.matrix([])

    temp1 = []
    #parpaer phi array
    for i in origin_x:
        #print(i)
        temp2 = []
        for j in range(M):
            temp2.append(sigmoid_func( (i-(4*j/M))/0.1 ))
        temp1.append(temp2)
    phi_j_of_x_matrix = np.matrix(temp1)
    print(phi_j_of_x_matrix.shape)
    #print(N)
    #SN_matrix = SN(np.matrix(phi_j_of_x))
    #mN_matrix = mN(np.matrix(phi_j_of_x), SN_matrix)

    print()
