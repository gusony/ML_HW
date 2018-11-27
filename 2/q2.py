import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math

raw_class_result = []
raw_feature = []
W_matrix = [] #(3,7)
phi_matrix = [] #(180, 7)
a_martix = [] #(3,180)
y_matrix = []
EW_list = []
EW_times = []
optimal_w_matrix =[]

def init_W_matrix():
    result = []
    for i in range(3): # 3 classes
        temp = []
        for j in range(7): # 7 features
            temp.append(0)
        result.append(temp)
    return(np.matrix(result)) #return (3,7)

def get_y_matrix(a):
    result = []
    for n in range(a.shape[1]):
        deno = 0
        for i in range(a.shape[0]):
            deno += exp(a.item(i,n))

        temp = []
        for k in range(a.shape[0]):
            temp.append(exp(a.item(k,n))/deno)
        result.append(temp)
    return(np.matrix(result,dtype=np.float64))


def error_func(t,y):
    return(sum((-1)*t.item(n,k)*np.log(float(y.item(n,k))) for n in range(t.shape[0]) for k in range(t.shape[1]) ) )


def degreeE(j,y,t):
    y_t = y-t
    sum_array = sum( y_t[n,j]*np.array(phi_matrix[n,:]) for n in range(y.shape[0]) )
    return(np.matrix(sum_array.flatten().tolist()))

def test_degreeE(y,t):
    phi = np.matrix(raw_feature)
    result = phi.T.dot(y-t)
    return(result.T)


def Hj(j,y,phi):
    y_1_y = np.array(y)*(1-np.array(y))
    result = sum( y_1_y[n,j]*phi[n,:].T.dot(phi[n,:]) for n in range(y.shape[0]) )
    return(result)


def update_w_mtx(y, w_old, phi, t):#w_old : matrix
    w_new = []

    for j in range(w_old.shape[0]): #for each class
        degreeE_matrix = test_degreeE(y,t)#degreeE(j,y,t)
        HJ = np.matrix(Hj(j,y,phi),dtype=np.float32)
        w_new_matrix = w_old[j,:] - degreeE_matrix.dot(pinv(HJ.T))
        w_new.append(np.array(w_new_matrix).flatten().tolist())
    #test_degreeE(y,t)
    return(np.matrix(w_new))


#read file
with open('train.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        raw_class_result.append(list(map(float, row[0:3])))
        raw_feature.append(list(map(float, row[3:10])))

W_matrix = init_W_matrix()
phi_matrix = np.matrix(raw_feature)
t_matrix = np.matrix(raw_class_result)

optimal_w_matrix = np.matrix( pinv(phi_matrix.T.dot(phi_matrix)).dot(phi_matrix.T).dot(t_matrix) )


for i in range(1): # test : run 100 time to update W
    print(i)
    a_martix = W_matrix.dot(phi_matrix.T)
    y_matrix = get_y_matrix(a_martix)
    print('a_matrix.dtype',a_martix.dtype,y_matrix.dtype)
    W_matrix = update_w_mtx(y_matrix, W_matrix, phi_matrix, t_matrix)
print(W_matrix,'\n--------------------------------\n')
#print('y_matrix',y_matrix.item(0,))
print('\noptimal_w_matrix:\n',optimal_w_matrix.T)

#part2
#print(error_func(t_matrix, y_matrix))
#part3
f,part3 = plt.subplots(sharex=True,sharey=True)
part3.hist(np.array(raw_feature)[:,0])
part3.hist(np.array(raw_feature)[:,1])
part3.hist(np.array(raw_feature)[:,2])
part3.hist(np.array(raw_feature)[:,3])
part3.hist(np.array(raw_feature)[:,4])
part3.hist(np.array(raw_feature)[:,5])
part3.hist(np.array(raw_feature)[:,6])


f,part5 = plt.subplots()
part5.scatter(np.array(raw_feature)[:,3], np.array(raw_feature)[:,6])











#plt.show()





