import csv  #read .csv
import numpy as np
#from numpy.linalg import pinv
import matplotlib.pyplot as plt
#from sklearn import svm
#import seaborn as sns; sns.set()
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Manager


def f(i, result, x, uk):
    result[i] = sub.T.dot(sub)
    print(result[i])

if __name__ == '__main__':
    '''
    data = np.matrix([[1,2,3],[4,5,6]])
    print(np.array(data[1]).flatten())
    '''


    #data1 = np.matrix([[1,7,2],[1,3,5]])
    #print(np.sum(data1,axis=0))
    #print(np.matrix(data1/2, dtype=np.int64).tolist())


    # data2 = np.matrix([[1,3,5],[2,4,6]])

    # print(data1[0])
    # print(data2[1])
    # manager = Manager()
    # jobs = []
    # result = manager.list([[],[]])
    # for i in range(2):
    #     p = Process(target=f, args=(i, result, data1[i,:], data2[i,:]))
    #     jobs.append(p)
    #     p.start()

    # for proc in jobs:
    #     proc.join()



    #print(f(data1[0],data1[1]))
    #print('result',result)
    #total = np.matrix([0,0,0])
    #for i in range(len(data)):
    #    total += data[i]
    #print(total )



##########################
#good use

#data1.append(np.random.randint(5, size=(3)).tolist())
