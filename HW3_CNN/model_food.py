import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

torch.cuda.set_device(0)
seed = 0
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

input_size = 128


def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), input_size, input_size, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (input_size, input_size))        # 将读取的图片缩放至 (128,128) 大小
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
            nn.Linear(512*4*4, 1024),       # 全连接层
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class NewClassifier(nn.Module):
    def __init__(self):
        super(NewClassifier, self).__init__()
        # torch.nn.Conv2d(in_channels,out_Channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),          # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),    # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),          # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),   # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),          # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),   # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),          # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),          # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4*4, 1024),       # 全连接层
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 11)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

"""
Training
"""
# # model = Classifier().cuda()
# # model = AlexNet().cuda()
# model = NewClassifier().cuda()
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# num_epoch = 45
# print('epoch start..')
#
# train_acc_curve = []
# train_loss_curve = []
# valid_acc_curve = []
# valid_loss_curve = []
#
# for epoch in range(num_epoch):
#     epoch_start_time = time.time()
#     train_acc = 0.0
#     train_loss = 0.0
#     val_acc = 0.0
#     val_loss = 0.0
#
#     model.train()       # 训练模型时会在前面使用
#     for i, data in enumerate(train_loader):
#         optimizer.zero_grad()
#         train_pred = model(data[0].cuda())      # 使用 model 得到预测的概率分布，实质上是去调用 model 中的 forward 函数
#         batch_loss = loss(train_pred, data[1].cuda())   # 计算 loss，(prediction 和 label 必须同时在 CPU 或是 GPU 上)
#         batch_loss.backward()       # 利用 back propagation 计算出每个参数的 gradient
#         optimizer.step()            # 以 optimizer 用 gradient 更新参数值
#
#         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         train_loss += batch_loss.item()
#
#     model.eval()        # 测试模型时在前面使用
#
#     with torch.no_grad():
#         for i, data in enumerate(val_loader):
#             val_pred = model(data[0].cuda())
#             batch_loss = loss(val_pred, data[1].cuda())
#
#             val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#             val_loss += batch_loss.item()
#
#         train_acc_curve.append(train_acc/train_set.__len__())
#         train_loss_curve.append(train_loss/train_set.__len__())
#         valid_acc_curve.append(val_acc/val_set.__len__())
#         valid_loss_curve.append(val_loss/val_set.__len__())
#
#         # print results
#         print('[%03d/%03d]  %2.2f sec(s)    Train Acc: %3.6f   Loss:   %3.6f   |   Val Acc:    %3.6f   Loss:   %3.6f' %
#               (epoch + 1, num_epoch, time.time()-epoch_start_time,
#                train_acc/train_set.__len__(),
#                train_loss/train_set.__len__(),
#                val_acc/val_set.__len__(),
#                val_loss/val_set.__len__()))
#
#
# x_axis = np.linspace(0, num_epoch, num_epoch)
# plt.plot(x_axis, train_acc_curve, label='train_acc')
# plt.plot(x_axis, train_loss_curve, label='train_loss')
# plt.plot(x_axis, valid_acc_curve, label='valid_acc')
# plt.plot(x_axis, valid_loss_curve, label='valid_loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()


"""
得到性能较好的模型参数之后，使用train set 和 valid set 共同训练
"""
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = NewClassifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
num_epoch = 50

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    print('[%03d/%03d]   %2.2f   sec(s)  Train Acc:  %3.6f   Loss:   %3.6f'  %
          (epoch + 1, num_epoch, time.time()-epoch_start_time,
           train_acc / train_val_set.__len__(),
           train_loss / train_val_set.__len__()))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# with open('predict.csv', 'w') as f:
#     f.write('Id, Category\n')
#     for i, y in enumerate(prediction):
#         f.write('{}, {}\n'.format(i, y))

with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

