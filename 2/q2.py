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
EW_list = []
x = []
EW_list = []
test_feature = []

ε = 0.5

#############################################################################
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
    return(np.matrix(result,dtype=np.float64))

def error_func(t,y):
    return(sum((-1)*t.item(n,k)*np.log(float(y.item(n,k))) for n in range(t.shape[0]) for k in range(t.shape[1]) ) )

def get_R_matrix(y):
    array_y = np.array(y)
    R = []
    for n in range(y.shape[0]):
        temp = []
        for j in range(y.shape[0]):
            if n == j:
                temp.append(array_y[n,:].dot(1-array_y[n,:]))
            else:
                temp.append(0)
        R.append(temp)
    return(np.matrix(R))

def Z(phi, w_old, R, y, t):
    return(phi.dot(w_old.T) - pinv(R).dot(y-t) )

def update_w_mtx(phi, w_old, t, y):
    R = get_R_matrix(y) # shape(n,n))
    z = Z(phi, w_old, R, y, t)
    return( pinv(phi.T.dot(R).dot(phi)).dot(phi.T).dot(R).dot(z).T  )

def get_class_result(y):
    y_list = np.array(y).tolist()
    result = np.matrix([[0,0,0] for n in range(y.shape[0])])
    for n in range(y.shape[0]):
        result.itemset((n,y_list[n].index(max(y_list[n]))), 1)
    return(np.matrix(result))
#############################################################################

#read file
with open('train.csv', newline='') as trainfile:
    rows = csv.reader(trainfile)
    for row in rows:
        raw_class_result.append(list(map(float, row[0:3])))
        raw_feature.append(list(map(float, row[3:10])))
with open('test.csv', newline='') as testfile:
    rows = csv.reader(testfile)
    for row in rows:
        test_feature.append(list(map(float,row)))
    test_feature = np.matrix(test_feature)

#set init value
W_matrix = init_W_matrix()
phi_matrix = np.matrix(raw_feature)
t_matrix = np.matrix(raw_class_result)

#compute W iteratively
i=1
while i:
    print(i)
    x.append(i)
    a_martix = W_matrix.dot(phi_matrix.T)
    y_matrix = get_y_matrix(a_martix)
    W_matrix = update_w_mtx(phi_matrix, W_matrix, t_matrix, y_matrix)
    EW_list.append(error_func(t_matrix, y_matrix))
    if EW_list[i-1] < ε:
        break
    i+=1


#part1
f,part1 = plt.subplots()
part1.set_title('HW2 Q2 part1, E(w)')
part1.set_xlabel('iteration times')
part1.text(round(i/2),170,"ε=0.5\ninteration times ="+str(i))
part1.scatter(x, EW_list)

#part2
class_result_matrix = get_class_result(get_y_matrix(W_matrix.dot(test_feature.T)))
class_result_list = np.array(class_result_matrix.dot(np.matrix([[1],[2],[3]]))).flatten().tolist()
f,part2 = plt.subplots()
part2.set_title('HW2 Q2 part2, show class result oftest data')
part2.scatter(list(range(class_result_matrix.shape[0])), class_result_list)

#part3
f,part3 = plt.subplots(sharex=True,sharey=True)
part3.set_title('HW2 Q2 part3')
part3.hist(np.array(raw_feature)[:,0],label='f[0]')
part3.hist(np.array(raw_feature)[:,1],label='f[1]')
part3.hist(np.array(raw_feature)[:,2],label='f[2]')
part3.hist(np.array(raw_feature)[:,3],label='f[3]')
part3.hist(np.array(raw_feature)[:,4],label='f[4]')
part3.hist(np.array(raw_feature)[:,5],label='f[5]')
part3.hist(np.array(raw_feature)[:,6],label='f[6]')
plt.legend()


f,part5 = plt.subplots()
part5.scatter(np.array(raw_feature)[:,3], np.array(raw_feature)[:,6])
part5.set_title('HW2 Q2 part5')










plt.show()





