# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:38:11 2018

@author: Eleam Emmanuel
"""

# importing the libraries
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # take all the columns but leave the last one(-1)
Y = dataset.iloc[:, 1].values

# splitting the dataset into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# feature scaling
"""sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)"""

# fitting simple linear regression to the training set                       
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# Visualizing the Training set result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
