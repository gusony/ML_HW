import csv
import numpy as np
from numpy.linalg import pinv
from sympy import *
import matplotlib.pyplot as plt


#init variable
train_x = [] #dtype : list, origin data from train.csv
train_y = []
test_x = []
test_y = []
train_no_tr = []
test_no_tr  = []
train_no_pr = []
test_no_pr  = []
train_no_mi = []
test_no_mi  = []
M1_X = []
M2_X = []
M3_X = []
M1_Y = []
M2_Y = []
M3_Y = []
LR = lambda vec_x, w: np.dot(vec_x,two_d_matrix_tolist(w))
#vec_x : list
#two_d_matrix_tolist(W1) :list
test_lamb = 0.1


def one_row_product(data_array, order): # each X product []
#return : [1,x1,x2,x3,  x1x1,x1x2,...,x3x3,  x1x1x1,x1x1x2...,x3x3x3]
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

def get_dm_for_part2(x_list, order):#design_matrix
  result = []
  for i in x_list:
    result.append(one_row_product(i, order))
  return (np.matrix(result))


def get_dm(data,order):#design_matrix
  result = []
  for i in data:#train_x:
    result.append(one_row_product(i, order))
  return (np.matrix(result))

#return list
def get_predY_matrix(order, w,x):
  result = []
  for i in x:
    result.append( [LR(one_row_product(i, order),w)] )
  return(np.matrix(result))

# 2-dimension matrix convert to 1-dimension array
# return list
def two_d_matrix_tolist(X):
  return(X.reshape(np.size(X)).tolist()[0])

def rms_error(pred_y, ture_y):
  E_W = 0.5 * np.sum(np.power(pred_y - ture_y , 2))
  E_rms = np.sqrt(2*E_W/len(pred_y))
  return E_rms

def reg_rms_error(pred_y, ture_y,lamda,w):
  E_W = 0.5 * np.sum(np.power(pred_y - ture_y , 2)) +  0.5*lamda*two_d_matrix_tolist(w.T*w)[0]
  E_rms = np.sqrt(2*E_W/len(pred_y))
  return E_rms

def get_lambIM(lamb, shape): #get_lamb_identity_matrix #lambda 單位矩陣
  result = []
  for i in range(shape):
    temp = []
    for j in range(shape):
      temp.append(lamb)
    result.append(temp)
  return(np.matrix(result))


# 開啟 CSV 檔案
with open('train.csv', newline='') as trainfile:
  rows = csv.reader(trainfile)
  for row in rows:
    train_no_tr.append([float(row[1]), float(row[2])])
    train_no_pr.append([float(row[0]), float(row[0])])
    train_no_mi.append([float(row[2]), float(row[0])])

    train_x.append([float(row[0]), float(row[1]), float(row[2])])
    train_y.append(float(row[3]))
  train_Y = np.mat([train_y]).T

with open('test.csv', newline='') as testfile:
  rows = csv.reader(testfile)
  for row in rows:
    test_no_tr.append([float(row[1]), float(row[2])])
    test_no_pr.append([float(row[0]), float(row[0])])
    test_no_mi.append([float(row[2]), float(row[0])])
    test_x.append([float(row[0]), float(row[1]), float(row[2])])
    test_y.append(float(row[3]))
  test_Y = np.mat([test_y]).T


#######################################################################
#part2 , M = 3
M3_train_no_tr = get_dm_for_part2(train_no_tr,3)
W3_train_no_tr = pinv(M3_train_no_tr) * train_Y
predY_no_tr = get_dm(test_no_tr,3)* W3_train_no_tr

M3_train_no_pr = get_dm_for_part2(train_no_pr,3)
W3_train_no_pr = pinv(M3_train_no_pr) * train_Y
predY_no_pr = get_dm(test_no_pr,3)* W3_train_no_pr

M3_train_no_mi = get_dm_for_part2(train_no_mi,3)
W3_train_no_mi = pinv(M3_train_no_mi) * train_Y
predY_no_mi = get_dm(test_no_mi,3)* W3_train_no_mi

part2_err_list =[rms_error(predY_no_tr,test_Y), rms_error(predY_no_pr,test_Y), rms_error(predY_no_mi,test_Y)]
f, part2_f = plt.subplots()
part2_f.scatter(['no_tr','no_pr','no_mi'],part2_err_list)


#######################################################################
# m=1
M1_X = get_dm(train_x,1) #(18576,4)
W1 = pinv(M1_X) * train_Y # W1 is M=1  W #shape (4,1)
R_W1 = pinv(get_lambIM(test_lamb, M1_X.shape[1]) + M1_X.T*M1_X)*M1_X.T*train_Y #(4,1)
predY_M1 = get_dm(test_x,1)* W1

########################################################################
# m=2
M2_X = get_dm(train_x,2) #(18576,13)
W2 = pinv(M2_X) * train_Y #(13,1)
R_W2 = pinv(get_lambIM(test_lamb, M2_X.shape[1]) + M2_X.T*M2_X)*M2_X.T*train_Y
predY_M2 = get_dm(test_x,2)* W2

########################################################################
# m=3
M3_X = get_dm(train_x,3)
W3 = pinv(M3_X) * train_Y
R_W3 = pinv(get_lambIM(test_lamb, M3_X.shape[1]) + M3_X.T*M3_X)*M3_X.T*train_Y
predY_M3 = get_dm(test_x,3)* W3


########################################################################
Erms_list =[rms_error(predY_M1,test_Y), rms_error(predY_M2, test_Y),rms_error(predY_M3, test_Y)]
print('Erms_list:',Erms_list)
print('-------------------------')

reg_rms_error_list_01 = [reg_rms_error(np.matrix(get_predY_matrix(1,R_W1, test_x)),test_Y, test_lamb, R_W1),reg_rms_error(np.matrix(get_predY_matrix(2,R_W2, test_x)),test_Y, test_lamb, R_W2),reg_rms_error(np.matrix(get_predY_matrix(3,R_W3, test_x)),test_Y, test_lamb, R_W3)]
print('reg_rms_error_list_01',reg_rms_error_list_01)
print('-------------------------')

test_lamb = 0.001
reg_rms_error_list_0001 = [reg_rms_error(np.matrix(get_predY_matrix(1,R_W1, test_x)),test_Y, test_lamb, R_W1),reg_rms_error(np.matrix(get_predY_matrix(2,R_W2, test_x)),test_Y, test_lamb, R_W2),reg_rms_error(np.matrix(get_predY_matrix(3,R_W3, test_x)),test_Y, test_lamb, R_W3)]
print('reg_rms_error_list_0001',reg_rms_error_list_0001)
##########################################################################
M = [1,2,3]
plt.title('E_RMS M=1 to M=3')
plt.xlabel('M(order)')
plt.ylabel('E_RMS')

# plt.scatter(M, Erms_list,c='b',label='Err(no '+u"\u03BB"+')')
# plt.scatter(M, reg_rms_error_list_01,c='r',marker='^',label='Reg Err('+u"\u03BB"+'=0.1)')
# plt.scatter(M, reg_rms_error_list_0001,c='g',marker='*',label='Reg Err('+u"\u03BB"+'=0.001)')


f, (erms, RErms01, RErms0001) = plt.subplots(1,3)
erms.scatter(M, Erms_list,c='b',label='Err(no '+u"\u03BB"+')')
RErms01.scatter(M, reg_rms_error_list_01,c='r',marker='^',label='Reg Err('+u"\u03BB"+'=0.1)')
RErms0001.scatter(M, reg_rms_error_list_0001,c='g',marker='*',label='Reg Err('+u"\u03BB"+'=0.001)')
plt.legend(loc='upper right')
plt.show()
