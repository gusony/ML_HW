# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math


a =  np.array([[1,2,],[3,4]])

print(a.flatten())
# mean = np.array([0, 0])
# cov = np.matrix([[1, 0], [0, 100]])
# print(mean.shape,cov.shape)
# x, y = np.random.multivariate_normal(mean, cov, 5000).T
# print(np.random.multivariate_normal(mean, cov, 5000).shape)
# print(x)
# print()

# print(y)
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()