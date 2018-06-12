# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:13:31 2018

@author: Eleam Emmanuel
"""

# Hierarchial clustering
# Importing the libraries

import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing the dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3,4]].values

# Using the dendogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# fitting Hirarchial cluster to the mall dataset
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
            s=100, c='red', label='Careful clients')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=100, c='blue', label='Standard clients')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=100, c='green', label='Target clients')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=100, c='cyan', label='Careless clients')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
            s=100, c='magenta', label='Sensible clients')
plt.title("Clusters of clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending score (1 - 100)")
plt.legend()
plt.show()
