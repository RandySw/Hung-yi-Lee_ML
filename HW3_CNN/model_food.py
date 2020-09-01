import os
import numpy as np
# import cv2
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow_core.python.keras import layers
from tensorflow_core.python.keras.api._v2.keras import optimizers


# import pandas as pd
# import time
# from keras.models import Sequential

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset


#%% Practice



#%% Read Image
def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


workspace_dir = './food-11'
print('Reading Data...')
train_x, train_y = readfile(os.path.join(workspace_dir, 'training'), True)
print('Size of training data = {}'.format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print('Size of validation data = {}'.format(len(val_x)))
test_x, test_y = readfile(os.path.join(workspace_dir, 'testing'), False)
print('Size of testing data = {)'.format(len(test_x)))

#%% Data
# training 时做data augmentation
#train_transform = transforms




