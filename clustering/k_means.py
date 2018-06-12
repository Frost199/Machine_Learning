# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:28:06 2018

@author: Eleam Emmanuel
"""

# This is to clear and reset and use CTRL+L in console to clear console
#%reset -f

# K-Means Clustering
# importing the libraries
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# importing the dataset with pandas
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3,4]].values

# Using the elbow method to find optimal number of clusters
# another name for wcss is inertia
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()

# applying k-means to the mall dataset
kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=100, c='red', label='Target clients')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=100, c='blue', label='Sensible clients')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=100, c='green', label='Standard clients')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
            s=100, c='cyan', label='Careful clients')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],
            s=100, c='magenta', label='Careless clients')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')
plt.title("Clusters of clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending score (1 - 100)")
plt.legend()
plt.show()
