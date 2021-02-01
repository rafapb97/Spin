import os
import time
import numpy as np

print("tensorflowShizz")
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, \
    Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical




# WORKING DIRECTORY #
#####################
print("makedir")
# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath((os.path.dirname(os.path.realpath(__file__))))
#os.makedirs(path_wd)

# GET DATASET #
###############
print("load mnist data")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input so we can train ANN with it.
# Will be converted back to integers for SNN layer.
x_train = x_train / 255
x_test = x_test / 255

# Add a channel dimension.
axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
x_train = np.expand_dims(x_train, axis)
x_test = np.expand_dims(x_test, axis)

# One-hot encode target vectors.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# SNN toolbox will not do any training, but we save a subset of the training
# set so the toolbox can use it when normalizing the network parameters.
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

# CREATE ANN #
##############

# This section creates a simple CNN using Keras, and trains it
# with backpropagation. There are no spikes involved at this point.

input_shape = x_train.shape[1:]
input_layer = Input(input_shape)

layer = Conv2D(filters=4,
               kernel_size=(3, 3),
               strides=(2, 2),
               activation='relu',
               use_bias=False)(input_layer)
"""
layer = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation='relu',
               use_bias=False)(layer)
#layer = AveragePooling2D()(layer)
layer = Conv2D(filters=8,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               use_bias=False)(layer)
"""
layer = Flatten()(layer)
layer = Dropout(0.01)(layer)

layer = Dense(units=10,
              activation='softmax',
              use_bias=False)(layer)


model = Model(input_layer, layer)

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Train model with backprop.
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2,
          validation_data=(x_test, y_test))

# Store model so SNN Toolbox can find it.
model_name = 'mnist_cnn'
keras.models.save_model(model, model_name + '.h5')