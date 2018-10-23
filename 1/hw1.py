import csv
import numpy as np
from numpy.linalg import pinv
from sympy import *


#init variable
train_x = [] #dtype : list, origin data from train.csv
train_y = []
test_x = []
test_y = []
LR = lambda vec_x, w: np.dot(vec_x,two_d_matrix_tolist(w))
test_lamb = 0.1


def one_row_product(data_array, order): # each X product []
#return : [1,x1,x2,x3, x1x1,x1x2,...,x3x3,x1x1x1,x1x1x2...,x3x3x3]
  result = [1]
  if order >=1:
    for i in data_array:
      result.append(i)
  if order >= 2:
    for i in data_array:
      for j in data_array:
        result.append(i*j)
  if order >= 3:
    for i in data_array:
      for j in data_array:
        for k in data_array:
          result.append(i*j*k)
  return (result)

def get_dm(order):#design_matrix
  result = []
  for i in train_x:
    result.append(one_row_product(i, order))
  return (np.matrix(result))

def get_predY_list(order, w):
  result = []
  for i in test_x:
    result.append( [LR(one_row_product(i, order),w)] )
  return(result)

def rms_error(pred_y, ture_y):
  E_W = 0.5 * np.sum(np.power(pred_y - ture_y , 2))
  E_rms = np.sqrt(2*E_W/len(pred_y))
  return E_rms

def reg_rms_error(pred_y, ture_y,lamda,w):
  E_W = 0.5 * np.sum(np.power(pred_y - ture_y , 2)) + 0.5* lamda*np.transpose(w)*w
  E_rms = np.sqrt(2*E_W/len(pred_y))
  return E_rms

def lamb_identity(lamb, shape):
  result = []
  for i in range(shape):
    temp = []
    for j in range(shape):
      temp.append(lamb)
    result.append(temp)
  return(np.matrix(result))

def two_d_matrix_tolist(X):
  temp = []
  for i in X.tolist():
    for j in i:
      temp.append(j)
  return(temp)

# 開啟 CSV 檔案
with open('train.csv', newline='') as trainfile:
  rows = csv.reader(trainfile)
  for row in rows:
    train_x.append([float(row[0]), float(row[1]), float(row[2])])
    train_y.append(float(row[3]))
  train_Y = np.mat([train_y]).T

with open('test.csv', newline='') as testfile:
  rows = csv.reader(testfile)
  for row in rows:
    test_x.append([float(row[0]), float(row[1]), float(row[2])])
    test_y.append(float(row[3]))
  test_Y = np.mat([test_y]).T


#######################################################################
# m=1
M1_X = get_dm(1)
W1 = pinv(M1_X) * train_Y # W1 is M=1  W
R_W1 = pinv(lamb_identity(test_lamb, M1_X.shape[1]) + M1_X.T*M1_X)*M1_X.T*train_Y

########################################################################
# m=2
M2_X = get_dm(2)
W2 = pinv(M2_X) * train_Y
R_W2 = pinv(lamb_identity(test_lamb, M2_X.shape[1]) + M2_X.T*M2_X)*M2_X.T*train_Y

########################################################################
# m=3
M3_X = get_dm(3)
W3 = pinv(M3_X) * train_Y
R_W3 = pinv(lamb_identity(test_lamb, M3_X.shape[1]) + M3_X.T*M3_X)*M3_X.T*train_Y

########################################################################
#LR = lambda vec_x, w: np.dot(vec_x,w)
test_data=[880,322,8.3252]
print( LR(one_row_product(test_data, 1), W1)) #測試一筆資料
print( LR(one_row_product(test_data, 1), R_W1))
print( LR(one_row_product(test_data, 2), W2))
print( LR(one_row_product(test_data, 2), R_W2))
print( LR(one_row_product(test_data, 3), W3))
print( LR(one_row_product(test_data, 3), R_W3))
print('-------------------------')
print(rms_error(np.matrix(get_predY_list(1,W1)),test_Y))
print(rms_error(np.matrix(get_predY_list(2,W2)),test_Y))
print(rms_error(np.matrix(get_predY_list(3,W3)),test_Y))
print('-------------------------')
print(reg_rms_error(np.matrix(get_predY_list(1,R_W1)),test_Y, test_lamb, R_W1))
print(reg_rms_error(np.matrix(get_predY_list(2,R_W2)),test_Y, test_lamb, R_W2))
print(reg_rms_error(np.matrix(get_predY_list(3,R_W3)),test_Y, test_lamb, R_W3))
