# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:15:02 2018

@author: Eleam Emmanuel
"""

# Kernel PCA
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # take all the columns but leave the last one(-1)
Y = dataset.iloc[:, 4].values

# splitting the dataset into a training set and a test set
# here we are using 100 observation which is 100/400 = 0.25, so test_size=0.25
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

# feature scaling
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# Applying Kernel PCA
k_pca = KernelPCA(n_components=2, kernel='rbf')
x_train = k_pca.fit_transform(x_train)
x_test = k_pca.transform(x_test)

# fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predicting the Test set resulrt
y_pred = classifier.predict(x_test)

# making the confusion matrix to test performance of our logistic_regression
cm = confusion_matrix(y_test, y_pred)

accuracy = print("Accuracy is {}%".format(((cm[0, 0] + cm[1, 1]) / len(y_test)) * 100))

# Visualizing the training set results
X_set, Y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualizing the test set results
X_set, Y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
