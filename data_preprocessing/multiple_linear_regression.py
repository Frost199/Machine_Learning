# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 07:41:23 2018

@author: Eleam Emmanuel
"""
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # take all the columns but leave the last one(-1)
Y = dataset.iloc[:, 4].values

# Encoding our categorical variables for X
label_encoding_X = LabelEncoder()
X[:, 3] = label_encoding_X.fit_transform(X[:, 3])
# in this example, we are not checking for greater or lesser
# in our encoding, we just want to give an identifiable
# encoding, so we will use 'OneHotEncoding', to give dummy 
# data with the index of the column you want to OneHotEncode
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# splitting the dataset into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
"""sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)"""

# fitting multiple linear regression to the training set
regressor = LinearRegression()
# fitting regressor to our training set
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# ===================================================================== #
# Building the optimal model using backward elimination using SL = 0.05 #
# ===================================================================== #

# add colums of ones to the matrix of features
# adding our matrix X to columns of 50 ones
# axis=1 for column and axis=0 for rows
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
 
# creating an optimal matirx of features, this X_optimal will only contain 
# independent variable that will have high impact on the profit i.e the 
# dependent variable
# seperating all the colums of X and putting it in X_optimal
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]

# fitting all possible predictors in our case X_optimal to our model
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary()

# x2 has a higher P value, so we remove it an fit again, and 
# x2 is equivalent 2 in line 62 
X_optimal = X[:, [0, 1, 3, 4, 5]]

# fitting all possible predictors in our case X_optimal to our model
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary()

# x1 has a higher P value, so we remove it an fit again, and 
# x1 is equivalent 1 in line 62 
X_optimal = X[:, [0, 3, 4, 5]]

# fitting all possible predictors in our case X_optimal to our model
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary()

# x2 has a higher P value, so we remove it an fit again, and 
# x2 is equivalent 4 in line 62 
X_optimal = X[:, [0, 3, 5]]

# fitting all possible predictors in our case X_optimal to our model
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary()

# x2 has a higher P value, so we remove it an fit again, and 
# x2 is equivalent 5 in line 62 
X_optimal = X[:, [0, 3]]

# fitting all possible predictors in our case X_optimal to our model
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
regressor_OLS.summary()
