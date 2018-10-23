import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def two_d_matrix_tolist(X):
  temp = []
  for i in X.tolist():
    for j in i:
      temp.append(j)
  return(temp)

'''
X = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[151,156,126]]
target_matrix = np.matrix([[13],[14],[15],[16],[20]])
X = np.matrix(X)
#print((X.T*X).I)
def lamb_identity(lamb, shape):
  result = []
  for i in range(shape):
    temp = []
    for j in range(shape):
      temp.append(lamb)
    result.append(temp)
  return(result)

#W  = inv(X.T*X)*X.T*target_matrix


W =  inv(lamb_identity(0.01,X.shape[1]) +X.T*X)*X.T*target_matrix
print(W)
'''
