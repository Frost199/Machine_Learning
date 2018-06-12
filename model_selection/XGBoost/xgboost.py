# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:54:53 2018

@author: Eleam Emmanuel
"""
# XGBoost
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # take all the columns but leave the last one(-1)
Y = dataset.iloc[:, 13].values

# Encoding our categorical variables
label_encoding_X_1 = LabelEncoder()
X[:, 1] = label_encoding_X_1.fit_transform(X[:, 1])
label_encoding_X_2 = LabelEncoder()
X[:, 2] = label_encoding_X_2.fit_transform(X[:, 2])

"""
    In this example, we are not checking for greater or lesser in our encoding, 
we just want to give an identifiable
encoding, so we will use 'OneHotEncoding', to give dummy data
categorical_features=[] points to the column you want to add dummy variable
we dont need to do one hot encoding because it just two conditions
(Male or Female), so if we try to avoid the dummy variable trap,
we will be left with one single column.
 """
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()

# avoiding the dummy variable trap after encoding column country
X = X[:, 1:]

# splitting the dataset into a training set and a test set
# here we are using 100 observation which is 100/400 = 0.25, so test_size=0.2
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

# fitting XGBoost to the Training set
classifier = XGBClassifier()

y_pred = classifier.predict(x_test)
# making the confusion matrix to test performance of our logistic_regression
cm = confusion_matrix(y_test, y_pred)

# Testing for accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / 2000

# Appplying K-Fold Cross Validation for a better model analysis 
# than confusion matrix
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=30)
accuracies.mean() # mean
accuracies.std()  # standard deviation