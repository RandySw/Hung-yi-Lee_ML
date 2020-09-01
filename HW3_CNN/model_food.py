import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time


def readfile(path, label):
    """
    loading image files and save as numpy array -> x
    :param path:
    :param label:   label flag
    :return:
            x -> image
            y -> label
    """
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split('_')[0])

        if label:
            return x, y
        else:
            return x

# Reading image
workspace_dir = './data/food-11'
print('Reading data')
train_x, train_y = readfile(os.path.join(workspace_dir, 'training'), True)
print('Size of validation data = {}'.format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print('Size of validation data = {}'.format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, 'testing'), False)
print('Size of testing data = {}'.format(len(test_x)))


# training 时做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(),        # 随机旋转图片
    transforms.ToTensor(),              # 将图片转为tensor，并且把数值 normalize 到 [0,1] (data normalization)
])

# testing 时不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transforms








