# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 06:03:08 2018

@author: Eleam Emmanuel
"""

import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
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

# fitting linear regression to the datatset
linear_reg = LinearRegression()
linear_reg.fit(X, Y)

# fitting polynomial regression to the dataset
# transform X to a polynomial of desired power
Polynomial_reg = PolynomialFeatures(degree=4)
X_polynomial = Polynomial_reg.fit_transform(X)
# now fitting our X_polynomial
Polynomial_reg.fit(X_polynomial, Y)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_polynomial, Y)

# Visualizing the linear regression result
plt.scatter(X, Y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, linear_reg_2.predict(Polynomial_reg.fit_transform(X_grid)), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# predicting a new result with linear regression
linear_reg.predict(6.5)

# predicting a new result with linear regression
linear_reg_2.predict(Polynomial_reg.fit_transform(6.5))