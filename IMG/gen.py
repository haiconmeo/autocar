
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
INPUT_NORMALIZATION = 255.0
OUTPUT_NORMALIZATION = 655.35 #picked this number to compare results with data source model.
img_shape = (66, 200, 3)
batch_size = 128
def generator(df, batch_size):
    img_list = df['img']
    wheel_axis = df['wheel-axis']    
    # create an empty batch
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))
    index = 0
    while True:
        for i in range(batch_size):
            label = wheel_axis.iloc[index]
            img_name = img_list.iloc[index]
            pil_img = image.load_img(path_to_data+img_name)
            # Data augmentation           
            if(np.random.choice(2, 1)[0] == 1):
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                label = -1 * label            
            batch_img[i] = image.img_to_array(pil_img)
            batch_label[i] = label
            index += 1
            if index == len(img_list):
                #End of an epoch hence reshuffle
                df = df.sample(frac=1).reset_index(drop=True)
                img_list = df['img']
                wheel_axis = df['wheel-axis']
                index = 0
        yield batch_img / INPUT_NORMALIZATION, (batch_label / OUTPUT_NORMALIZATION)