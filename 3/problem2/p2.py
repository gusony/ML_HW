import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm


def readdata(textname):
    with open(textname, newline='') as rowfile:
        return ([row for row in csv.reader(rowfile)])
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

def calcEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T * (os.X*os.X[k,:].T) + os.b)
    Ek = fXk - float(os.labelMat[k])
    return Ek

'''
函數功能：將輸入的元素限定在一個範圍內
'''
def clipAlpha(input, Low, high):
    if input>high:
        input = high
    if input<Low:
        input = Low

    return input

'''
函數功能：在輸入的參數i和m之間隨機選擇一個不同於i的數字，也就是在選定了i之後隨機選取一個與之配對的alpha的取值的下標
'''
def selectJrand(i, m):
    j = i
    while j==i:
        j = int(np.random.uniform(0, m))

    return j
'''
函數功能：選擇一個SMO算法中與外層配對的alpha值的下標
'''
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:  #啓發式選取配對的j，計算誤差
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
'''
SMO算法中的優化部分
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or \
        ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>oS.C)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        '''公式(7)'''
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if H == L:
            #print ("H==L program continued")
            return 0

        '''公式（8）（9）'''
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        if 0 <= eta:
            #print ("eta>=0 program continued")
            return
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            #print ("j not moving enough %s" % ("program continued"))
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i) #更新誤差緩存'

        '''設置常數項 b '''
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i] and oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j] and oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

'''
完整版SMO算法
dataMatIn: 訓練數據
classLabels: 數據標籤
C: 常量
toler: 容錯度
maxIter: 最大迭代次數
kTup=('lin', 0): 核函數類型
'''
def SMOP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged>0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                #print ("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0) * (oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                #print ("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (0==alphaPairsChanged):
            entireSet = True
        print ("iteration number: %d" % (iter))
    return oS.b, oS.alphas


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



clf.fit(np.delete(train_x,range(100,150),0), np.delete(train_t,range(100,150),0))
flt[0,1].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='r')
clf.fit(np.delete(train_x,range(50,100),0), np.delete(train_t,range(50,100),0))
flt[0,1].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='b')
clf.fit(np.delete(train_x,range(0,50),0), np.delete(train_t,range(0,50),0))
flt[0,1].plot(x, (-clf.coef_[0][0]*x + clf.intercept_[0])/clf.coef_[0][1], color='g')


#b, alphas = SMOP(train_x, train_t, 1, 0.001, 100)
#print("\nb:\n",b)
#print("\nalphas:\n",len(alphas))

#print(clf.support_vectors_.shape)
#print(clf.n_support_)
#print("clf.dual_coef_",clf.dual_coef_)
#print("clf.coef_\n",clf.coef_)
#print("clf.intercept_\n",clf.intercept_)
#print(clf.decision_function(train_x))

#test_t = [1 if train_t[i] !=2 else 2 for i in range(len(train_t))]

poly_svc = svm.SVC(kernel='poly', degree=2 ,gamma='auto',decision_function_shape='ovr')
poly_svc.fit(train_x, train_t)
#print(poly_svc.dual_coef_.shape)
print(poly_svc.support_)
print(poly_svc.n_support_)
print(poly_svc.support_vectors_)
#print(poly_svc.intercept_.shape)
#print()

#for i in range(len(train_t)):
    #print(train_x[i],poly_svc.support_vectors_[i])









plt.show()