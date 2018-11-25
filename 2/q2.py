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
    return(np.matrix(result))


def error_func(t,y):
    return(sum((-1)*t.item(n,k)*np.log(float(y.item(n,k))) for n in range(t.shape[0]) for k in range(t.shape[1]) ) )

#read file
with open('train.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        raw_class_result.append(list(map(float, row[0:3])))
        raw_feature.append(list(map(float, row[3:10])))

W_matrix = init_W_matrix()
phi_matrix = np.matrix(raw_feature)
a_martix = W_matrix.dot(phi_matrix.T)
y_matrix = get_y_matrix(a_martix)
t_matrix = np.matrix(raw_class_result)
print(error_func(t_matrix, y_matrix))
print(error_func2(t_matrix, y_matrix))



