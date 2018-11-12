# ML HW2
import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg
import math

print(10**-6*np.eye(2))


a = np.matrix([[1,2],[3,4]])
b = np.matrix([[5,6],[7,8]])
print(a.dot(b))
print (c)

#mean = [0, 0]
#cov = [[1, 0], [0, 100]]  # diagonal covariance
#x, y = np.random.multivariate_normal(mean, cov, 5000).T
#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()