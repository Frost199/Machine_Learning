# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:48:00 2018

@author: USER
"""

# PART 1 - Building the CNN

# importing the keras model and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Inializing the CNN
classifier = Sequential()

# step 1 - Convolution
# we can use 64, or 128, 256 for the (32, (3, 3))
# for theano backend, input_shape=(64, 64, 3) is input_shape=(3, 64, 64)
# using relu activation function for non_linearity
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3),
                             activation='relu'))

# step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolution layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step - 3 Flattening
classifier.add(Flatten())

# step 4 - Full Connection
# change output_dense to units an update in Kera API
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# PART 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, 
                                   zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         # number of images in our training set
                         steps_per_epoch=8000, epochs=50,
                         validation_data=test_set,
                        # no of images in our test set
                        validation_steps=2000)
