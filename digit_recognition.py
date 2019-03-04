# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:44:59 2019

@author: intel
"""
#import packages
import keras 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

## Import MNIST Dataset 
from keras.datasets import mnist

## load the data into the respective train and test datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

## Train
print ("shape of X_train", X_train.shape)
print ("type of X_train", type(X_train))

## Test
print ("shape of X_test", X_test.shape)
print ("type of X_test", type(X_test))

#adding no of color layers and bormalizing the input
X_train = (X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')) / 255
X_test = (X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')) / 255


## Convers 1-D class array to 10-D "one hot encoded" array
Y_train = keras.utils.to_categorical(Y_train,10)
Y_test = keras.utils.to_categorical(Y_test,10)


## Import models and layers from Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

# Create a Sequential Model  
model = Sequential()
model.add(Conv2D(filters = 30, kernel_size = 5, input_shape=(28, 28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate = 0.2))
## Flatten Layer 
model.add(Flatten())
## Dense Layer with 128 neurons 
model.add(Dense(units = 128, activation='relu'))
## Dropout Layer with a different rate of 50% 
model.add(Dropout(0.5))
## Final Layer 
model.add(Dense(10, activation='softmax'))

## Compile the model 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

## Fit the model on the training data
model.fit(X_train, Y_train, epochs=10, verbose=2)

## Evaluate the model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

