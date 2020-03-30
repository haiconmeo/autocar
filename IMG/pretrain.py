import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
data_df = pd.read_csv(os.path.join(os.getcwd(),  'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
X = data_df[['center', 'left', 'right']].values

y = data_df['steering'].values

def load_image(data_dir, image_file):
    """
    Đọc ảnh RGB từ file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))
# def load_image(image_file):
  
#     return mpimg.imread(image_file)

def crop(image):
  
    return image[60:-25, :, :]


def resize(image):
  
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
  
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Pre-process ảnh
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

# img = load_image("E:/IMG/IMG/center_2020_03_23_08_50_21_301.jpg")
# img = preprocess(img)
# cv2.imshow("image", img)
# cv2.waitKey(0)
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Trả về ảnh và góc lái tương ứng cho việc training
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
           
            image = load_image(data_dir, center) 
            
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

