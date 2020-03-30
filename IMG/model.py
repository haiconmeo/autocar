import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten,MaxPool2D

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

model = Sequential()

from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/

class  model_train:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))
        model.summary()
        return model
