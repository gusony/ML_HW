import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans

max_interaction = 1000

#input data, 2D [n data, features].
#return k data of center[number of k, features]
def keans(x, k_num):
    #init k
    uk = [x[np.random.randint(len(x))] for i in range(k_num)]

    interaction = 0
    labels = [] # result

    while interaction <= max_interaction:
        print('interaction:',interaction,'\n-------------------')
        for n in range(len(x)): #對於每一筆資料
            labels.append(uk.indexOf(min([(x[n]-center)**2 for center in uk]))) #計算到每個群中心的距離 取最小的群中心




img = cv2.imread('hw3.jpg')

X = img.flatten()
print(X.shape)
#print(type(img[0][0]))
#print(img[0])
#print(img.shape)
#print(img[0,0])
#cv2.imshow('my',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#kmeans = KMeans(2, random_state=0)
#print(kmeans.fit(X))
#labels = kmeans.fit(X).predict(X)
#for i in labels:
    #print(labels)
#print(type(labels))
#print(labels)
#col = np.array([[100,50,3] if labels[i] == 0 else [124,27,59] for i in range(labels.shape[0])])
#col = col.astype(np.uint8)
#print(col.dtype)
#print(type(col), col.reshape(246,480,3).shape)
newimg = col.reshape(246,480,3)
#print(newimg)
cv2.imshow('test',newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(img.shape[0], img.shape[1],labels.shape)
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        plt.scatter(i, j,c=col)#, c=labels,  cmap=['white','black']);
#plt.show()