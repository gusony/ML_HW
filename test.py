import csv  #read .csv
import numpy as np
#from numpy.linalg import pinv
import matplotlib.pyplot as plt
#from sklearn import svm
#import seaborn as sns; sns.set()
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Manager


def f(i, result):
    result[i] = i
    print(result[i])

if __name__ == '__main__':


    #data = [0.]*10

    manager = Manager()
    jobs = []
    result = manager.list([0.]*10)
    for i in range(10):
        p = Process(target=f, args=(i, result))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(result)



##########################
#good use

#data1.append(np.random.randint(5, size=(3)).tolist())
