# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:49:35 2018

@author: Eleam Emmanuel
Topic: Data Preprocessing
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #take all the columns but leave the last one(-1)
Y = dataset.iloc[:, 3].values

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3]) # fitting the second column and third column we use three sice the upper bound is exculded
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding our categorical variables
label_encoding_X = LabelEncoder()
X[:, 0] = label_encoding_X.fit_transform(X[:, 0])

# in this example, we are not checking for greater or lesser in our encoding, we just want to give an identifiable
# encoding, so we will use 'OneHotEncoding', to give dummy data
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

# Encoding our categorical variables for Y
label_encoding_Y = LabelEncoder()
Y = label_encoding_Y.fit_transform(Y)

# splitting the dataset into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
