# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:36:55 2020

@author: OjaswitaNegi
"""
import tensorflow as tf
print(tf.__version__)

            
mnist=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(x_train[45])
print(y_train[45])
print(x_train[45])

#normalize
x_train=x_train/255.0
x_test=x_test/255.0

model=tf.keras.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(128,activation=tf.nn.relu), tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)

model.evaluate(x_test,y_test)

classifications=model.predict(x_test)
print(classifications[0])
