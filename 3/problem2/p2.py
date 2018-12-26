import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm


def readdata(textname):
    with open(textname, newline='') as rowfile:
        return ([row for row in csv.reader(rowfile)])

def poly_kernal(xi,xj):
    return((np.inner(xi,xj))**2)

def ovo_linear_and_plot(start, end, str_color):
    x = np.linspace(-1, 1, 1000)
    clf = svm.LinearSVC()
    clf.fit(np.delete(train_x,range(start,end),0), np.delete(train_t,range(start,end),0))
    flt[0,1].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color=str_color)


train_x = np.array(readdata("x_train.csv"), dtype=float)
train_t = np.array(readdata("t_train.csv"), dtype=int).flatten()

f,flt = plt.subplots(2,2,sharex='all',sharey='all')
flt[0,0].set(xlim=[-1, 1], ylim=[-1, 1])
flt[0,0].set_title('Linear SVC (ovr)')
flt[0,1].set_title('Linear SVC (ovo)')
flt[1,0].set_title('Poly SVC (degree=2,ovr)')
flt[1,1].set_title('Poly SVC (degree=2,ovo)')
for i in range(2):
    for j in range(2):
        for n in range(len(train_t)):
            if train_t[n] ==   1 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='r')
            elif train_t[n] == 2 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='b')
            elif train_t[n] == 3 :
                flt[i,j].scatter(train_x[n,0], train_x[n,1], color='g')


#clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
clf = svm.LinearSVC()
clf.fit(train_x, train_t)
# one vs rest
x = np.linspace(-1, 1, 1000)
flt[0,0].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='r')
flt[0,0].plot(x, (-clf.coef_[1][0]*x + clf.intercept_[1])/clf.coef_[1][1], color='b')
flt[0,0].plot(x, (-clf.coef_[2][0]*x + clf.intercept_[2])/clf.coef_[2][1], color='g')

#one vs one
ovo_linear_and_plot(100, 150, 'r')
ovo_linear_and_plot(50, 100, 'b')
ovo_linear_and_plot(0, 50, 'g')


poly_svc = svm.SVC(kernel=poly_kernal)
temp_t = [1 if train_t[i]!=2 else 2 for i in range(len(train_t))]
poly_svc.fit(train_x, temp_t)
#w =

print(poly_svc.dual_coef_)
print(poly_svc.support_)
print(poly_svc.n_support_)
print(poly_svc.support_vectors_)
print(poly_svc.intercept_)

poly_svc.dual_coef_[0],train
#print()

#for i in range(len(train_t)):
    #print(train_x[i],poly_svc.support_vectors_[i])


#print(clf.support_vectors_.shape)
#print(clf.n_support_)
#print("clf.dual_coef_",clf.dual_coef_)
#print("clf.coef_\n",clf.coef_)
#print("clf.intercept_\n",clf.intercept_)
#print(clf.decision_function(train_x))







plt.show()