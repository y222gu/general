# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:18:58 2021

@author: 123
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('siRNA_data.csv')

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method for siRNA')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.savefig('elbow method for siRNA')

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset)
cluster_1 = np.reshape(np.where(y_kmeans == 0),(501,1))
cluster_2 = np.reshape(np.where(y_kmeans == 1),(631,1))

#plt.scatter(dataset[cluster_1, 0], dataset[cluster_1, 1], s = 10, c = 'red', label = 'Cluster 1')
#plt.scatter(dataset[cluster_2, 0], dataset[cluster_2, 1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(dataset[cluster_1, 1], dataset[cluster_1, 2], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(dataset[cluster_2, 1], dataset[cluster_2, 2], s = 10, c = 'blue', label = 'Cluster 2')


#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'k', label = 'Centroids')
plt.title('Clusters of siRNA LNPs')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

