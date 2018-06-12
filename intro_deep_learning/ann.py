# -*- coding: utf-8 -*-
"""
Created on Sat May  5 23:29:37 2018

@author: USER
"""
# Artificial neural network 

# Part 1 - Data Preprocessing
import pandas as pd
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from ann_visualizer.visualize import ann_viz

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

# feature scaling is very important and should not be skipped in 
# deep learning because of the high computation required
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
# ************************************************************************* #
# ************************************************************************* #

# Step 2 _ making the Artificial Neural Network (ANN)
# Initializing the ANN
classifier = Sequential()

# Adding the inout layer and the first hidden layer
# for the output_dim = number of independent + number of dependent / 2
# for this example output_dim = 11 + 1 / 2 = 6
# This below is deperecated in keras 2.0
# classifier.add(Dense(output_dim=6, init='uniform', activation='relu',
#                     input_dim=11))
classifier.add(Dense(activation="relu", input_dim=11, units=6, 
                     kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, 
                     kernel_initializer="uniform"))

# compiling the Neural Network using sochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)
# ************************************************************************* #
# ************************************************************************* #

# Step 3, making the predictions and evaluating the model
# predicting the Test set result
y_pred = classifier.predict(x_test)
# using a true or false just to check our y_pred to see which
# customer leaves and stay amd a threshold of 0.5
y_pred = (y_pred > 0.5)

# making the confusion matrix to test performance of our logistic_regression
cm = confusion_matrix(y_test, y_pred)

# Testing for accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / 2000

# Visualizing the ANN
ann_viz(classifier, view=True, filename="network.gv", 
        title="Churn for a Bank")