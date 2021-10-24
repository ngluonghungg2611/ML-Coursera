import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[-2,-1]].values
# print(X)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.legend()
plt.show()
'''
    Áp dụng phương pháp khửu tay thì nhận thấy
    => Nhận thấy n_cluster = 5 là hợp lý nhất
'''
# kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
# y_kmeans = kmeans.fit_predict(X)
# print(y_kmeans)
# #Visualising the cluster
# plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0,1], s = 100, c='red', label='cluster 1')
# plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1,1], s = 100, c='green', label='cluster 2')
# plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2,1], s = 100, c='blue', label='cluster 3')
# plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3,1], s = 100, c='yellow', label='cluster 4')
# plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4,1], s = 100, c='purple', label='cluster 5')
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'orange', label='centroid')
# plt.title('Cluster of client')
# plt.xlabel('Annual income($)')
# plt.ylabel('Spending score')
# plt.legend()
# plt.show()