import csv
import numpy as np
from numpy.linalg import pinv
from sympy import *
import matplotlib.pyplot as plt
import numpy.linalg

def gauss(A):
    m = len(A)
    print(m)
    assert all([len(row) == m + 1 for row in A[1:]]), "Matrix rows have non-uniform length"+len(row)
    n = m + 1

    for k in range(m):
        pivots = [abs(A[i][k]) for i in range(k, m)]
        i_max = pivots.index(max(pivots)) + k

        # Check for singular matrix
        assert A[i_max][k] != 0, "Matrix is singular!"

        # Swap rows
        A[k], A[i_max] = A[i_max], A[k]


        for i in range(k + 1, m):
            f = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= A[k][j] * f

            # Fill lower triangular matrix with zeros:
            A[i][k] = 0

    # Solve equation Ax=b for an upper triangular matrix A
    x = []
    for i in range(m - 1, -1, -1):
        x.insert(0, A[i][m] / A[i][i])
        for k in range(i - 1, -1, -1):
            A[k][m] -= A[k][i] * x[0]
    return x

def get_lambIM(lamb, shape): #get_lamb_identity_matrix #lambda 單位矩陣
  result = []
  for i in range(shape):
    temp = []
    for j in range(shape):
      temp.append(lamb)
    result.append(temp)
  return(np.matrix(result))

test = [[1,2],[3,4],[6,7],[5,7],[1,2]]
temp = get_lambIM(1,2)
print(numpy.linalg.solve(test, temp))
print(pinv(test))