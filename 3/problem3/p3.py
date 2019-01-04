import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
import seaborn as sns; sns.set()
from scipy.stats import multivariate_normal
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Manager
from datetime import datetime as dt


test_k_list = [2,3,5]#,20]
uk_list = []
eps = 0.000001
max_interaction = 20


#input data, 2D [n data, features].
#return k data of center[number of k, features]
def keans(x, k_num):
    #init k
    uk = np.array([list(x[np.random.randint(len(x))].data) for i in range(k_num)])
    last_uk = np.array([[],[]])
    interaction = 0

    while interaction < max_interaction:
        interaction += 1
        γ = np.zeros((x.shape[0],k_num))

        #距離平方和
        ds = np.argmin(np.sum(np.array([(x[:]-center)**2 for center in uk ]), axis=2), axis=0)
        for n in range(x.shape[0]):
            γ[n,ds[n]] = 1

        #update uk
        uk = np.array([  γ.T.dot(x)[k,:] / np.sum(γ,axis=0)[k]  for k in range(k_num)])

        #uk 收斂
        if np.array_equal(uk, last_uk):
            break
        last_uk = uk

    labels = np.array([n for n in ds])
    return(labels, uk.astype(np.uint8) , γ)

def Nk(labels, k):
    unique, counts = np.unique(labels, return_counts=True)
    data = dict(zip(unique, counts))
    return data[k]
def Σk(labels, k):
    Nk(labels,k)
    πk = Nk(labels,k)/labels.shape[0]

def f(i, result, X, μ, Σ):
    result[i] = sub.T.dot(sub)
'''
def f2(result, ind, n, K, π, Yn, mu, cov):
    temp = 0
    for k in range(K):
        temp += π[k] * myNfunc(Yn,mu[k], cov[k])
        #R[:, k] = πk[k] * multivariate_normal.pdf(X, mean=np.array(μ[i]).flatten(), cov=Σ[i])
        #print('temp=',temp)
    result[n] = np.log(np.sum(temp))

def myNfunc(Xn, μk, Σk, d=3):
    a = ( (2*np.pi)**d * np.linalg.det(Σk) ) ** -.5 # exp 前面的分數
    b = -.5 * (Xn - μk).dot(inv(Σk)).dot((Xn - μk).T) #exp 裡面的值
    return(a*np.exp(b))
'''
DEBUG = True

def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)

def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

def getExpectation(Y, μ, cov, π):
    N = Y.shape[0]
    K = π.shape[0]

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # 响应度矩阵，行对应样本，列对应响应度
    γ = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, μ[k], cov[k])
    prob = np.mat(prob)


    # 计算每个模型对每个样本的响应度
    for k in range(K):
        γ[:, k] = π[k] * prob[:, k]
    for i in range(N):
        γ[i, :] /= np.sum(γ[i, :])
    return γ

def maximize(Y, γ):
    N, D = Y.shape
    K = γ.shape[1]

    #初始化参数值
    μ = np.zeros((K, D))
    cov = []
    π = np.zeros(K)
    Nk = np.array(np.sum(γ, axis=0)).reshape(K)

    # 更新每个模型的参数
    for k in range(K):
        # 更新 μ
        μ[k,:] = np.sum(γ[:, k].T.dot(Y), axis=0) / Nk[k]

        # 更新 cov
        cov_k = np.mat(np.zeros((D, D)))
        ysubuk  = Y - μ[k]# y subtract u[k]
        a = np.mat(np.array(γ[:,k])*np.array(ysubuk))
        cov_k = ( a.T.dot(ysubuk)/Nk[k])
        cov.append(cov_k)
        # 更新 π
        π[k] = Nk[k] / N
    cov = np.array(cov)
    return μ, cov, π

def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y

def init_params(shape, K, γ, uk):
    N, D = shape
    μ = uk/255 #np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    π = np.sum(γ,axis=0)/np.sum(γ)
    #debug("Parameters initialized.")
    #debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")
    return μ, cov, π

def GMM_EM(init_Y, K, times, γ,uk):
    Y = scale_data(init_Y)
    μ, cov, π = init_params(Y.shape, K, γ, uk)
    log_likelihoods = []
    for i in range(times):
        a = dt.now()
        γ = getExpectation(Y, μ, cov, π)
        print('getExpectation time',dt.now() - a)
        a = dt.now()
        μ, cov, π = maximize(Y, γ)
        #debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")
        print('maximize time',dt.now() - a)

        R = []
        for k in range(K):
            ymuk = Y - μ[k]
            mat_dia = np.einsum('ij,ji->i',ymuk.dot(cov[k]), ymuk.T )
            a = ((2*np.pi)**Y.shape[1] * np.linalg.det(cov[k])) ** -.5

            R.append(π[k] * a * np.exp(-.5 * mat_dia) )
        R = np.array(R)
        log_likelihood = np.sum(np.log(R))
        print('log_likelihood:',log_likelihood)
        log_likelihoods.append(log_likelihood)



    #debug("{sep} Result {sep}".format(sep="-" * 20))
    #debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")
    #return mu, cov, π
    return log_likelihoods


#readfile
img = cv2.imread('hw3.jpg')
matY = np.matrix(img.reshape(img.shape[0]*img.shape[1],img.shape[2]),dtype=np.float64)
f,flt = plt.subplots(4,sharex=True, sharey=True)
xaxis = np.arange(1,max_interaction+1)
for k_num in test_k_list:
    print('k_num=',k_num)
    labels, uk, γ= keans(img.reshape(img.shape[0]*img.shape[1],img.shape[2]), k_num)
    uk_list.append(uk)
    new_img = np.array([uk[x] for x in labels])
    new_img = new_img.reshape(246,480,3)



    # eps = 0.000001
    # gmm = GMM(k, eps, max_iters)
    # params = gmm.fit_EM(matY,γ)
    log_likelihoods = GMM_EM(matY, k_num, max_interaction, γ,uk)
    flt[test_k_list.index( k_num )].scatter(xaxis, log_likelihoods)


    #print(Nk(labels, 1))
    #cv2.imshow('test',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imwrite("k="+str(k_num)+".jpg", new_img)

plt.show()