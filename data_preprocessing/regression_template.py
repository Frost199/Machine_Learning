# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:37:28 2018

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 06:03:08 2018

@author: Eleam Emmanuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# take all the columns but leave the last one(-1)
# always make sure our independent variable is a matrix not a vector and 
# dependent variable can be a vector
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values
                
# splitting the dataset into a training set and a test set
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
"""sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train)"""

# fitting the regression Model to the dataset

# Create your regressor here

# predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the regression result
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the regression result (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()