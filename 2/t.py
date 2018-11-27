# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


a = [[1,2,3,4],[-5,-6,-7,-8]]
print(min(a[0]))
print(min(a[1]))
print(a[1].index(min(a[1])))

#print(a[0,:].min(1))
#n=0
#a.itemset((n,a[n,:].index(a[n,:].min(1))), 0)
#print(a)


# def get_R_matrix(y):
#     array_y = np.array(y)
#     R = []
#     for n in range(y.shape[0]):
#         temp = []
#         for j in range(y.shape[0]):
#             if n == j:
#                 temp.append(array_y[n,:].dot(1-array_y[n,:]))
#             else:
#                 temp.append(0)
#         R.append(temp)
#     return(np.matrix(R))

#a = np.matrix([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
#b = np.matrix([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
#c = np.matrix([[100,100,100,100,100,100,100],[100,100,100,100,100,100,100],[100,100,100,100,100,100,100]])
#print(a)
#print(b)
#print(c - a*b)
#print(get_R_matrix(np.matrix([[1,2,3],[4,5,6]])))

# phi = np.matrix([[1,2,3],[4,5,6]])
# temp = phi.T
# R = get_R_matrix(np.matrix([[1,2,3],[4,5,6]]))
# temp = temp.dot(R)
# temp = temp.dot(phi)
# print(temp)
# print(phi.T.dot(R).dot(phi))
