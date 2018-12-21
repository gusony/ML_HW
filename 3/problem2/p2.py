import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm


def readdata(textname):
    with open(textname, newline='') as rowfile:
        return ([row for row in csv.reader(rowfile)])

train_x = np.array(readdata("x_train.csv"), dtype=float)
train_t = np.array(readdata("t_train.csv"), dtype=int).flatten()

f,flt = plt.subplots(2,2,sharex='all',sharey='all')
flt[0,0].set(xlim=[-1, 1], ylim=[-1, 1])


for i in range(2):
    for j in range(2):
        for n in range(len(train_t)):
            if train_t[n] ==   1 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='r')
            elif train_t[n] == 2 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='b')
            elif train_t[n] == 3 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='g')

flt[0,0].set_title('LinearSVC ovr')
#clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
clf = svm.LinearSVC()
clf.fit(train_x, train_t)
#print(clf.support_vectors_.shape)
#print(clf.n_support_)
#print("clf.dual_coef_",clf.dual_coef_)
print("clf.coef_\n",clf.coef_)
print("clf.intercept_\n",clf.intercept_)
#print(clf.decision_function(train_x))

# one vs rest
x = np.linspace(-1, 1, 1000)
flt[0,0].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='r')
flt[0,0].plot(x, (-clf.coef_[1][0]*x + clf.intercept_[1])/clf.coef_[1][1], color='b')
flt[0,0].plot(x, (-clf.coef_[2][0]*x + clf.intercept_[2])/clf.coef_[2][1], color='g')





#for i  in range(len(train_t)):
#    if train_t[i] != 2:
#        train_t[i] = 1
# print(train_t)
# print([1 if train_t[i] !=2 else 2 for i in range(len(train_t))])

# clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
# clf.fit(train_x, np.array([1 if train_t[i] !=2 else 2 for i in range(len(train_t))]))
# print("clf.coef_\n",clf.coef_)
# print("clf.intercept_\n",clf.intercept_)

# flt[1].set(xlim=[-1, 1], ylim=[-1, 1])
# flt[0].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='pink')

# for n in range(len(train_t)):
#     if train_t[n] ==1 :
#         flt[1].scatter(train_x[n][0], train_x[n][1], color='r')
#     elif train_t[n] ==2 :
#         flt[1].scatter(train_x[n,0], train_x[n,1], color='b')










plt.show()