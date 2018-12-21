# ML HW3 problem1
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


train_x = []
train_t = []
test_x = []
test_t = []
C_matrix_list = []
θ_list = [[1,4,0,0],
        [0,0,0,1],
        [1,4,0,5],
        [1,64,10,0]] # const


def knm(θ, xn, xm):
    return(θ[0]*np.exp(-θ[1]*(xn-xm)**2/2)+θ[2]+θ[3] * xn * xm)

def CN(θ, x): # hyper parameter index
    return(np.matrix([ [knm(θ, x[i], x[j]) if i!=j else knm(θ, x[i], x[j])+1 for j in range(len(x))]for i in range(len(x)) ]))

def k_matrix(θ, xn): #return type = np.matrix
    return(np.matrix([knm(θ, train_x[i], xn) for i in range(60)]))

def mean_of_x(θ, k, CN_inv):
    return( k.dot(CN_inv).dot(np.matrix(train_t).T).item((0,0))  )

def var_matrix(θ, xn, k, CN_inv):
    c = knm(θ, xn, xn)+1
    return( (c - k.dot(CN_inv).dot(k.T)).item((0,0)) )

#read data
with open('gp.csv', newline='') as rowfile:
    rows = csv.reader(rowfile)
    i = 0
    for row in rows:
        if i < 60:
            train_x.append(float(row[0]))
            train_t.append(float(row[1]))
        else:
            test_x.append(float(row[0]))
            test_t.append(float(row[1]))
        i+=1

f,flt = plt.subplots(1,4)
flt[0].set_title('(1,4,0,0)')
flt[1].set_title('(0,0,0,1)')
flt[2].set_title('(1,4,0,5)')
flt[3].set_title('(1,64,10,0)')

C_matrix_list = [CN(θ, train_x) for θ in θ_list]

#bluild four C matrix
for θ in θ_list:
    k_list = [k_matrix(θ, test_x[i]) for i in range(len(test_x)) ]
    CN_inv = pinv(C_matrix_list[θ_list.index(θ)])

    mean_t_of_x = [mean_of_x(θ, k_list[i], CN_inv) for i in range(len(test_x))]
    cvar_t_of_x = [var_matrix(θ, test_x[i], k_list[i], CN_inv)     for i in range(len(test_x))]
    pluse_one = [ mean_t_of_x[i]+np.sqrt(cvar_t_of_x[i]) for i in range(len(test_x))]
    minus_one = [ mean_t_of_x[i]-np.sqrt(cvar_t_of_x[i]) for i in range(len(test_x))]

    #zip and sort together
    a = list(zip(test_x,mean_t_of_x, pluse_one, minus_one))
    a = sorted(a, key=lambda l:l[0],reverse=False )

    #mean curve
    flt[θ_list.index(θ)].plot([a[i][0] for i in range(len(a))], [a[i][1] for i in range(len(a))] ,color='r')
    #fill between pluse and minus one std area
    flt[θ_list.index(θ)].fill_between([a[i][0] for i in range(len(a))], [a[i][2] for i in range(len(a))], [a[i][3] for i in range(len(a))] ,color='pink')
    #pin out train data
    flt[θ_list.index(θ)].scatter(train_x, train_t, color='b')
    #computer Erms for train and test data
    Erms_train = round(np.sqrt((sum([(mean_t_of_x[i] - train_t[i])**2 for i in range(60)]))/60),5)
    Erms_test = round(np.sqrt((sum([(mean_t_of_x[i] - test_t[i])**2 for i in range(60)]))/60),5)
    flt[θ_list.index(θ)].text(0.4,0.9, 'Erms\ntrain:'+str(Erms_train)+"\ntest:"+str(Erms_test) ,transform=flt[θ_list.index(θ)].transAxes)#將文字顯示在subplots裡面的相對座標
plt.show()


