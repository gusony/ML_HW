# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
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
    return(1./(1.+np.exp(-x)))

def get_phi_matrix(x_list): #return np.matrix
    result = []
    for i in x_list:
        temp = []
        for j in range(M):
            temp.append(sigmoid_func( (i-(4*j/M))/0.1 ))
        print( ((i-(4*j/M))/0.1 ), temp[j])
        result.append(temp)
    return(np.matrix(result))

def SN(phi_matrix):
    #phi_matrix (N, 7)
    #result     (7, 7)
    return(pinv(phi_matrix.T.dot(phi_matrix) + (10**-6)*np.eye(7)))

def mN(phi_matrix,SN,t_list):
    #phi_matrix.T (7, N)
    #t_list.T     (7, 1)
    #SN           (7, 7)
    #return       (7, 1)
    return(SN.dot(phi_matrix.T.dot(np.matrix(t_list).T)))
################################################################################

#read data
with open('1_data.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        Raw_data_x.append(float(row[0]))
        Raw_data_t.append(float(row[1]))

#init plot
#for question 1-2:
f,part1_2 = plt.subplots(1,4)


for N in N_list:
    phi_j_of_x_matrix = np.matrix([])
    SN_matrix = np.matrix([])
    mN_matrix = np.matrix([])
    sample_x = []
    sample_t = []
    part1_2[N_list.index(N)].set_title('N='+str(N))

    for i in range(N): # data set size [10,15,30,80]
        sample_x.append(Raw_data_x[i])
        sample_t.append(Raw_data_t[i])

    phi_j_of_x_matrix = get_phi_matrix(sample_x)
    SN_matrix = SN(phi_j_of_x_matrix) #print(np.array(SN_matrix.shape)) # (7,7)
    mN_matrix = mN(phi_j_of_x_matrix, SN_matrix, sample_t)

    mN_list = np.array(mN_matrix).flatten()



    x = np.arange(min(Raw_data_x)-1, max(Raw_data_x)+1, 0.01)
    for i in range(5): # 5 curve
        #np.poly1d : input W, return linear function
        line = np.poly1d(np.random.multivariate_normal(mN_list, SN_matrix).T)
        part1_2[N_list.index(N)].plot(x, line(x), '-')
        part1_2[N_list.index(N)].plot(sample_x, sample_t, '.')

plt.show()


print()




#reference
# reshape : https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.reshape.html
# flatten : https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.flatten.html
# multivariate_normal : https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal
# poly1d : https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.poly1d.html#numpy.poly1d
# arange : https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html
# matplotlib.pyplot.subplots : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
# matplotlib.pyplot.scatter : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html





