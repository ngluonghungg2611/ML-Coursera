import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
y = dataset.iloc[:, 1].values

# plt.scatter(X[:, 0], X[:,1], c = 'r')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
'''
    method= 'ward': giảm thiểu sự khác biệt giữa mỗi cụm
'''
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()
'''
    Trong khoảng distance dài nhất thì có 5 clusters -> 5 clusters là hợp lý nhất
'''
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s = 100, c ='yellow', label = 'Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s = 100, c = 'green', label = 'Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s = 100, c = 'orange', label = 'Sensible')
plt.title('Cluster of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
