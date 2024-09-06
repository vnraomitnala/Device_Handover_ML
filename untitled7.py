# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:48:31 2023

@author: Vijaya
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import plot_model

# Create a Sequential model
model = Sequential()

# Add Conv1D layer with 32 filters, kernel size 3, and input shape (input_length, input_dim)
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)))

# Add MaxPooling1D layer with pool size 2
model.add(MaxPooling1D(pool_size=2))

# Flatten the output
model.add(Flatten())

# Add a Dense layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu'))

# Add the output layer with appropriate number of neurons and activation function
model.add(Dense(1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Plot the model architecture and save it to a file
plot_model(model, to_file='1d_cnn_model.png', show_shapes=True, show_layer_names=True)


