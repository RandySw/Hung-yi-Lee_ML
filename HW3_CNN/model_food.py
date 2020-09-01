import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time

torch.cuda.set_device(0)


def readfile(path, label):
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
print('Size of training data = {}'.format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print('Size of validation data = {}'.format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, 'testing'), False)
print('Size of testing data = {}'.format(len(test_x)))

# training 时做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),            # 将 array 转换成 image
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),        # 随机旋转图片
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
        # label is required to be a LongTensor -> long type
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print('train/val loader finished.')


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels,out_Channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),    # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),   # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),   # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# Training
model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30
print('epoch start..')

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()       # 训练模型时会在前面使用
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()        # 测试模型时在前面使用

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # print results
        print('[%03d/%03d]  %2.2f sec(s)    Train Acc: %3.6f   Loss:   %3.6f   |   Val Acc:    %3.6f   Loss:   %3.6f' %
              (epoch + 1, num_epoch, time.time()-epoch_start_time,
               train_acc/train_set.__len__(),
               train_loss/train_set.__len__(),
               val_acc/val_set.__len__(),
               val_loss/val_set.__len__()))


"""
得到性能较好的模型参数之后，使用train set 和 valid set 共同训练
"""
model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()


