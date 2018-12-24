#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
import functions from other file
'''
import SVM_DataCreate as data_create

'''==============================================================================================
完整的SMO SVM算法
'''

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
            print "H==L program continued"
            return 0

        '''公式（8）（9）'''
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        if 0 <= eta:
            print "eta>=0 program continued"
            return
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough %s" % ("program continued")
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
                print "fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0) * (oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif (0==alphaPairsChanged):
            entireSet = True
        print "iteration number: %d" % (iter)
    return oS.b, oS.alphas


'''
函數功能：由計算出來的alphas獲得進行分類的權重向量
alphas: 計算出來的alpha向量
dataArr: 訓練數據
classLabels: 數據標籤
'''
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels)
    rows, cols = np.shape(dataArr)
    w = np.mat(np.zeros((cols, 1)))
    for i in range(rows):
        w += np.multiply(alphas[i, :]*labelMat[:, i], X[i,:].T)
    return w

'''
main function
'''
Data, DataLabel = data_create.GenerateData(10)     #generate data
print Data
b, alphas = SMOP(Data, DataLabel, 0.6, 0.001, 50)

SuportPoint = []
for i in range(np.shape(alphas)[0]):
    if 0 < alphas[i]:
        SuportPoint.append([i, alphas[i]])
SuportPoint = np.mat(np.array(SuportPoint))
print SuportPoint
weight_vector = calcWs(alphas, np.mat(Data), np.mat(DataLabel))
print weight_vector

#fig, ax =plt.subplots(1,2,figsize=(10,5))
plt.figure("decsion boundry")
plt.title("decsion boundry")
p1 = plt.subplot(121)
p2 = plt.subplot(122)
for i in range(Data.shape[0]):
    if 1 == DataLabel[i]:
        p1.plot(Data[i][0], Data[i][1], 'r^')
    if -1 == DataLabel[i]:
        p1.plot(Data[i][0], Data[i][1], 'gs')
p1.set_xlabel("X label")
p1.set_ylabel("Y label")
x = np.arange(-10, 10, 1)
l = float( weight_vector[0, :])
k = np.multiply(x, l)
y = np.multiply(weight_vector[0, :], x) + weight_vector[1, :] + b
y = np.array(y)
p1.plot(x, y[0, :], 'b')
p1.grid(True)

x = np.arange(0, 10, 0.5)
l = float( weight_vector[0, :])
k = np.multiply(x, l)
y = np.multiply(weight_vector[0, :], x) + weight_vector[1, :] + b
y = np.array(y)
p2.plot(x, y[0, :], 'b')
p2.set_xlabel("X label")
p2.set_ylabel("Y label")
p2.grid(True)
plt.title("input data distribution")
'''
SaveImg = raw_input("保存圖片(0), 不保存圖片(1)")
if "0" == SaveImg:
    try:
        plt.savefig('result.png', format='png')
    except Exception, e:
        print "save Image Error: %s" % (e)
elif "1" == SaveImg:
    print "result image has not been saved"
else:
    print "Input Error, please input '1' or '0' "
'''
plt.show()

print b
print alphas



