# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math

Raw_data_x = []
Raw_data_t = []
N_list = [10]#,15,30,80]
M = 7
s = 0.1

################################################################################
def sigmoid_func(x):
    return(1/(1+math.exp(x)))

def SN(phi_matrix):
    #phi_matrix (N, 7)
    #result_sn  (7, 7)
    result_sn = phi_matrix.T * phi_matrix
    result_sn = result_sn + 10**-6*np.eye(result_sn.shape[0])
    return(result_sn)

def mN(phi_matrix,SN,x_list):
    #phi_matrix.T (7, N)
    #x_list.T     (7, 1)
    #SN           (7, 7)
    #return       (7, 1)
    return(SN.dot(phi_matrix.T * np.matrix(x_list).T ))

def get_phi_matrix(x_list): #return np.matrix
    temp1 = []
    for i in x_list:
        temp2 = []
        for j in range(M):
            temp2.append(sigmoid_func( (i-(4*j/M))/0.1 ))
        temp1.append(temp2)
    return(np.matrix(temp1))
################################################################################

#read data
with open('1_data.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        Raw_data_x.append(float(row[0]))
        Raw_data_t.append(float(row[1]))

for N in N_list:
    phi_j_of_x_matrix = np.matrix([])
    SN_matrix = np.matrix([])
    mN_matrix = np.matrix([])

    sample_x = []
    for i in range(N):
        sample_x.append(Raw_data_x[i])
    phi_j_of_x_matrix = get_phi_matrix(sample_x)
    SN_matrix = SN(np.matrix(phi_j_of_x_matrix))
    mN_matrix = mN(np.matrix(phi_j_of_x_matrix), SN_matrix, sample_x)

    print()
