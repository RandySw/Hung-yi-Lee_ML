import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# Linear Module
# d_in = 3
# d_out = 4
# linear_module = nn.Linear(d_in, d_out)


# Sequential
d_in = 3
d_hidden = 4
d_out =1
model = torch.nn.Sequential(
                            nn.Linear(d_in, d_hidden),
                            nn.Tanh(),
                            nn.Linear(d_hidden, d_out),
                            nn.Sigmoid()
                            )

# Loss function
mse_loss_fn = nn.MSELoss()
input = torch.tensor([[0., 0., 0.]])
target = torch.tensor([[1., 0., -1]])

loss = mse_loss_fn(input, target)
print(loss)

print(torch.cuda.is_available())