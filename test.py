import csv  #read .csv
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns; sns.set()


data = [[1,2],[4,5],[6,7],[8,9]]
a = np.array([-3,-4])
b = np.array([5,6])
print(sum(a**2))
#print(np.random.randint(len(data)))

#a  = 10

#while(a++ < 10)
#    print(a)
# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=400, centers=4,
#                        cluster_std=0.60, random_state=0)

# X = X[:, ::-1] # flip axes for better plotting
# print(X)

# from sklearn.cluster import KMeans
# kmeans = KMeans(4, random_state=0)
# labels = kmeans.fit(X).predict(X)
# col = ['b' if labels[i] == 0 else 'r' for i in range(labels.shape[0])]
# print(X.shape,labels.shape)
# plt.scatter(X[:, 0], X[:, 1], c=col)#, s=40, cmap='viridis');
# plt.show()