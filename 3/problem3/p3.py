import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
import seaborn as sns; sns.set()
from scipy.stats import multivariate_normal
from multiprocessing import Pool

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


'''
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)
def getExpectation(Y, mu, cov, alpha):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = alpha.shape[0]

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
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma
def maximize(Y, gamma):
    # 樣本數跟特徵值
    N, D = Y.shape
    # 模型數
    K = gamma.shape[1]

    #初始化參數值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # 更新每個模型的参数
    for k in range(K):
        # 第 k 個模型對所有样本的响应度之和
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
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha
def GMM_EM(Y, K, times, flt):
    Y = scale_data(Y)
    print(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        print('\ntimes:',i,'\n')
        gamma = getExpectation(Y, mu, cov, alpha)

        mu, cov, alpha = maximize(Y, gamma)
        pi = [np.sum(gamma[:,i])/118080 for i in range(K)]
        log_likelihoods = (np.log(np.sum([k*multivariate_normal(mu[i],cov[j]).pdf(X) for k,i,j in zip(pi,range(mu.shape[0]),range(cov.shape[0]))])))
        print(log_likelihoods)
    return mu, cov, alpha

class GMM:

    def __init__(self, k = 3, eps = 0.0001):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`

        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple

    def fit_EM(self, X, max_iters = 1000):

        # n = number of data-points, d = dimension of data points
        n, d = X.shape

        # randomly choose the starting centroids/means
        ## as 3 of the points from datasets
        mu = X[np.random.choice(n, self.k, False), :]

        # initialize the covariance matrices for each gaussians
        Sigma= [np.eye(d)] * self.k

        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))

        ### log_likelihoods
        log_likelihoods = []

        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) )

        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iters:

            # E - Step

            ## Vectorized implementation of e-step equation to calculate the
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))

            log_likelihoods.append(log_likelihood)

            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T

            ## The number of datapoints belonging to each gaussian
            N_ks = np.sum(R, axis = 0)


            # M Step
            ## calculate the new mean and covariance for each gaussian by
            ## utilizing the new responsibilities
            for k in range(self.k):

                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])

                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))

                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break

        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
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
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)
'''

def f(i, result, X, μ, Σ):
    result[i] = sub.T.dot(sub)

def f2(i,result, xn, uk, sigmak):


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
        # Nk[k] = number of x belong to k cluster

        ########################################################################
        #init
        γ = init_γ
        n, d = X.shape
        Nk = np.sum(init_γ,axis=0) # return [53542,64538]
        πk = Nk/118080             # return [0.4534.., 0.54656...]

        μ = [ (np.matrix(γ.T[0].dot(X))/Nk[i]).flatten().tolist()[0] for i in range(self.k)]
        μ = np.array(μ, dtype=np.int64) # return [[34,86,106],[28,72,88]]

        # initialize the covariance matrices for each gaussians
        Σ= [np.eye(d)] * self.k
        print('Σ',Σ[0].shape)

        '''
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        γ = init_γ#np.zeros((n, self.k))

        # log_likelihoods
        log_likelihoods = []
        R = np.zeros((n, self.k))
        '''

        '''
        P = lambda μ, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - μ, np.dot(np.linalg.inv(s) , (X - μ).T).T ) )
        '''

        ########################################################################
        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iters:

            # E - Step
            for n in range(X.shape[1]):
            γ

            ## Vectorized implementation of e-step equation to calculate the
            ## membership for each of k -gaussians
            # for k in range(self.k):
            #     R[:, k] = πk[k] * P(μ[k], Σ[k])
            #     print('P',P(μ[k], Σ[k]))
            #print('scipy', multivariate_normal.pdf(x, mean=[0, 1], cov=[5, 2]))

            log_likelihood = 0
            ### Likelihood computation
            for n in range(X.shape[0]) :
                log_likelihood += np.sum(np.log( [ πk[i]*multivariate_normal.pdf(X[n], mean=np.array(μ[i]).flatten(), cov=Σ[i]) for i in range(self.k) ]))
            print('log_likelihood',log_likelihood)
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

#readfile
img = cv2.imread('hw3.jpg')
matY = np.matrix(img.reshape(img.shape[0]*img.shape[1],img.shape[2]),dtype=np.float64)
f,flt = plt.subplots(4)
i=0
for k_num in test_k_list:
    print( 'k_num=',k_num)
    labels, uk, γ= keans(img.reshape(img.shape[0]*img.shape[1],img.shape[2]), k_num)
    print('γ=\n',γ,γ.shape,'\n')
    uk_list.append(uk)
    new_img = np.array([uk[x] for x in labels])
    new_img = new_img.reshape(246,480,3)

    #P = lambda μ, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) * np.exp(-.5 * np.einsum('ij, ij -> i', X - μ, np.dot(np.linalg.inv(s) , (X - mu).T).T ) )
    #log_likelihood = np.sum(np.log(np.sum(γ, axis = 1)))

    k = 2
    max_iters = 10
    eps = 0.000001
    gmm = GMM(k, eps, max_iters)
    params = gmm.fit_EM(matY,γ)

    #mu, cov, alpha = GMM_EM(matY, k_num, 5, flt[i])
    #print(Nk(labels, 1))
    #cv2.imshow('test',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imwrite("k="+str(k_num)+".jpg", new_img)
    i+=1
#print(uk_list[2])
plt.show()