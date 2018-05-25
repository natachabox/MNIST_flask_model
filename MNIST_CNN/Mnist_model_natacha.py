# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:41:54 2018
# =============================================================================
# Natacha Cahorel
# Formation developpeur DATA IA Microsoft
# =============================================================================

@author: natacha
"""


print(__doc__)


import keras

from keras.datasets import mnist
from keras import models
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential


print(K.image_data_format())


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

input_shape = (28, 28, 1)
train_images = train_images.reshape(train_images.shape[0], input_shape[0], input_shape[1], 1)
test_images = test_images.reshape(test_images.shape[0], input_shape[0], input_shape[1], 1)

train_images = train_images.astype('float32')
train_images = train_images / 255
test_images = test_images.astype('float32')
test_images = test_images / 255

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

network = models.Sequential()
network.add(Conv2D(32, kernel_size=(3, 3), activation ='relu', input_shape = input_shape))
network.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
network.add(MaxPooling2D(pool_size = (2,2)))
network.add(Flatten())
network.add(Dense(128, activation='relu' ))
network.add(Dense(10, activation = "softmax"))

print(network.summary())

network.compile(loss = keras.losses.categorical_crossentropy, 
                optimizer = keras.optimizers.adadelta(), 
                metrics = ['accuracy'])

network.fit(train_images, train_labels, batch_size= 218, 
            epochs = 4,
            verbose = 1,
            validation_data= (test_images, test_labels))

score = network.evaluate(test_images, test_labels, verbose = 0)


network.save('mnist_natacha_model.h5')




