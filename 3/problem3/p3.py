import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans

test_k_list = [2,3,5,20]
max_interaction = 10

#input data, 2D [n data, features].
#return k data of center[number of k, features]
def keans(x, k_num):
    #init k
    uk = np.array([list(x[np.random.randint(len(x))].data) for i in range(k_num)])
    last_uk = np.array([[],[]])
    interaction = 0
    while interaction <= max_interaction:
        interaction += 1
        print('interaction:',interaction,'\n-------------------')
        γ = np.zeros((x.shape[0],k_num))

        #距離平方和
        ds = np.argmin(np.sum(np.array([np.array((x[:]-center)**2) for center in uk ]), axis=2), axis=0)
        for n in range(x.shape[0]):
            γ[n,ds[n]] = 1

        #update uk
        uk = np.array([  γ.T.dot(x)[k,:] / np.sum(γ,axis=0)[k]  for k in range(k_num)])

        #uk 收斂
        if np.array_equal(uk, last_uk):
            break
        last_uk = uk

    labels = np.array([n for n in ds])
    return(labels, uk.astype(np.uint8) )


#readfile
img = cv2.imread('hw3.jpg')
for k_num in test_k_list:
    print( 'k_num=',k_num)
    labels, uk = keans(img.reshape(img.shape[0]*img.shape[1],img.shape[2]), k_num)
    new_img = np.array([uk[x] for x in labels])
    new_img = new_img.reshape(246,480,3)

    #cv2.imshow('test',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite("k="+str(k_num)+".jpg", new_img)
print()