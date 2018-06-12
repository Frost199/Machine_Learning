# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:15:51 2018

@author: Eleam Emmanuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# take all the columns but leave the last one(-1)
# always make sure our independent variable is a matrix not a vector and 
# dependent variable can be a vector
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values
                
# splitting the dataset into a training set and a test set
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling without test and training
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# fitting SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

# predicting a new result, inverting the feature scaling
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

# Visualizing the regression result
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the SVR result (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (SVR Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()