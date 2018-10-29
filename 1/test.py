import csv
import numpy as np
from numpy.linalg import pinv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg



def get_lambIM(lamb, shape): #get_lamb_identity_matrix #lambda 單位矩陣
  result = []
  for i in range(shape):
    temp = []
    for j in range(shape):
      temp.append(lamb)
    result.append(temp)
  return(np.matrix(result))

def get_ext_matrix(shape):
    temp = []
    for i in range(shape):
        temp2 = []
        for j in range(shape):
            if i == j:
                temp2.append(1)
            else:
                temp2.append(0)
        temp.append(temp2)
    return(np.array(temp))


a = np.array([[1, 2], [2, 2]])
aa = np.array([[3, 2], [2, 2]])
print(2*get_ext_matrix[2])
#print(a*aa)
#print(np.dot(a,aa))



# test = [[2,3,4],[5,6,7],[2,9,10]]
# test = np.matrix(test)
# inv_matrix = numpy.linalg.solve(test, get_ext_matrix(3))
# print(inv_matrix)
# print( np.dot(inv_matrix.tolist() , test.tolist()))
# print(inv_matrix * test)
# #temp = get_lambIM(1,2)
# #print(numpy.linalg.solve(test, temp))
# #print(pinv(test))