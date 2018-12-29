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


test_k_list = [2]#,3,5]#,20]
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

DEBUG = True

######################################################
# 调试输出函数
# 由全局变量 DEBUG 控制输出
######################################################
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


######################################################
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
######################################################
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


######################################################
# E 步：计算每个模型对样本的响应度
# Y 为样本矩阵，每个样本一行，只有一个特征时为列向量
# mu 为均值多维数组，每行表示一个样本各个特征的均值
# cov 为协方差矩阵的数组，π 为模型响应度数组
######################################################
def getExpectation(Y, mu, cov, π):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = π.shape[0]

    # 为避免使用单个高斯模型或样本，导致返回结果的类型不一致
    # 因此要求样本数和模型个数必须大于1
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # 响应度矩阵，行对应样本，列对应响应度
    gamma = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = π[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


######################################################
# M 步：迭代模型参数
# Y 为样本矩阵，gamma 为响应度矩阵
######################################################
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    #初始化参数值
    mu = np.zeros((K, D))
    cov = []
    π = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        # 更新 cov
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
        # 更新 π
        π[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, π


######################################################
# 数据预处理
# 将所有数据都缩放到 0 和 1 之间
######################################################
def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


######################################################
# 初始化模型参数
# shape 是表示样本规模的二元组，(样本数, 特征数)
# K 表示模型个数
######################################################
def init_params(shape, K, γ, uk):
    N, D = shape
    mu = uk/255 #np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    π = np.sum(γ,axis=0)/np.sum(γ)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")
    return mu, cov, π


######################################################
# 高斯混合模型 EM 算法
# 给定样本矩阵 Y，计算模型参数
# K 为模型个数
# times 为迭代次数
######################################################
def GMM_EM(init_Y, K, times, γ,uk):
    Y = scale_data(init_Y)
    mu, cov, π = init_params(Y.shape, K, γ, uk)
    log_likelihoods = []
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, π)
        mu, cov, π = maximize(Y, gamma)
        #debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")

        for ind in range(int(Y.shape[0]/16)):
            #print('ind',ind)
            manager = Manager()
            ll_list = manager.list([0.]*16)
            jobs = []
            log_likelihood = 0

            for n in range(16):
                p = Process(target=f2, args=(ll_list, ind, n, K, π, Y[ind*n],mu, cov))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            log_likelihood += sum(ll_list)
        print(log_likelihood)
        log_likelihoods.append(log_likelihood)
        print(log_likelihoods)

        #     #print('n=',n)
        #     temp = 0
        #     for k in range(K):
        #         # print('k',k)
        #         # print('πk,',πk[k])
        #         # print('μ[k],',μ[k])
        #         # print('Σ[k],',Σ[k])
        #         # print('myNfunc,',myNfunc(X[n],μ[k], Σ[k]))
        #         temp += π[k] * myNfunc(Y[n],mu[k], cov[k])
        #         #R[:, k] = πk[k] * multivariate_normal.pdf(X, mean=np.array(μ[i]).flatten(), cov=Σ[i])
        #         print('temp=',temp)
        #     log_likelihood += np.log(np.sum(temp))
        # print('ll,',log_likelihood)


    #debug("{sep} Result {sep}".format(sep="-" * 20))
    #debug("mu:", mu, "cov:", cov, "π:", π, sep="\n")
    #return mu, cov, π
    return log_likelihoods


'''
class GMM:

    def __init__(self, k = 3, eps = 0.0001, max_iters = 1000):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`
        self.max_iters = max_iters

        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple

    def fit_EM(self, X, init_γ):

        # n = number of data-points,
        # d = dimension of data points
        # Nk[k] = number of x belong to k cluster # return [53542,64538]
        # πk = Nk/118080                          # return [0.4534.., 0.54656...]
        # μ = shape(k,d)                          # return [[34,86,106],[28,72,88]]
        ########################################################################
        #init
        γ = init_γ
        n, d = X.shape
        Nk = np.sum(init_γ,axis=0)
        πk = Nk/118080

        μ = [ (np.matrix(γ.T[0].dot(X))/Nk[i]).flatten().tolist()[0] for i in range(self.k)]
        μ = np.array(μ, dtype=np.int64) # return [[34,86,106],[28,72,88]]

        # initialize the covariance matrices for each gaussians
        Σ= [np.eye(d)] * self.k
        print('Σ',Σ[0].shape)


        # log_likelihoods
        log_likelihoods = []
        R = np.zeros((n, self.k))



        P = lambda μ, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i', X - μ, np.dot(pinv(s) , (X - μ).T).T ) )


        ########################################################################
        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iters:

            # E - Step

            log_likelihood = 0
            for n in range(X.shape[0]):
                print('n=',n)
                temp = 0
                for k in range(self.k):
                    # print('k',k)
                    # print('πk,',πk[k])
                    # print('μ[k],',μ[k])
                    # print('Σ[k],',Σ[k])
                    # print('myNfunc,',myNfunc(X[n],μ[k], Σ[k]))
                    temp += πk[k] * myNfunc(X[n],μ[k], Σ[k])
                    #R[:, k] = πk[k] * multivariate_normal.pdf(X, mean=np.array(μ[i]).flatten(), cov=Σ[i])
                    print('temp=',temp)
                log_likelihood += np.log(np.sum(temp))
            print('ll,',log_likelihood)
            log_likelihoods.append(log_likelihood)


            ## Normalize so that the responsibility matrix is row stochastic
            #γ = (γ.T / np.sum(γ, axis = 1)).T

            ## The number of datapoints belonging to each gaussian
            N_ks = np.sum(γ, axis = 0)


            # M Step
            ## calculate the new mean and covariance for each gaussian by
            ## utilizing the new responsibilities
            for k in range(self.k):

                ## means
                μ[k] = 1. / Nk[k] * np.sum(γ[:, k] * X.T, axis = 1).T
                x_μ = np.matrix(X - μ[k])

                ## covariances
                Σ[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_μ.T,  γ[:, k]), x_μ))

                ## and finally the probabilities
                πk[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break

        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['μ', 'Σ', 'πk', 'log_likelihoods', 'num_iters'])
        self.params.μ = μ
        self.params.Σ = Σ
        self.params.πk = πk
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)

        return self.params

    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()

    def predict(self, x):
        p = lambda μ, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - μ , \
                        np.dot(np.linalg.inv(s) , x - μ)))
        probs = np.array([πk * p(μ, s) for μ, s, πk in \
            zip(self.params.μ, self.params.Σ, self.params.πk)])
        return probs/np.sum(probs)
'''
#readfile
img = cv2.imread('hw3.jpg')
matY = np.matrix(img.reshape(img.shape[0]*img.shape[1],img.shape[2]),dtype=np.float64)
f,flt = plt.subplots(4)
i=0
for k_num in test_k_list:
    print('k_num=',k_num)
    labels, uk, γ= keans(img.reshape(img.shape[0]*img.shape[1],img.shape[2]), k_num)
    uk_list.append(uk)
    new_img = np.array([uk[x] for x in labels])
    new_img = new_img.reshape(246,480,3)

    #P = lambda μ, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i', X - μ, np.dot(np.linalg.inv(s) , (X - mu).T).T ) )
    #log_likelihood = np.sum(np.log(np.sum(γ, axis = 1)))

    # k = 2
    # max_iters = 1
    # eps = 0.000001
    # gmm = GMM(k, eps, max_iters)
    # params = gmm.fit_EM(matY,γ)

    mu, cov, π = GMM_EM(matY, k_num, 3, γ,uk)
    print()
    #print(Nk(labels, 1))
    #cv2.imshow('test',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imwrite("k="+str(k_num)+".jpg", new_img)

#plt.show()