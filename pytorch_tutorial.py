import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import numpy as np

"""
The tensor is similar to numpy's ndarray
"""


"""
tensor.view() is similarly to numpy.reshape()
"""
# N, C, W, H = 10000, 3, 28, 28
# X = torch.randn(N, C, W, H)
#
# print(X.shape)
# print(X.view(N, C, 784).shape)
# print(X.view(-1, C, 784).shape)     # automatically choose the 0th dimension


"""
Computation graphs
"""
# a = torch.tensor(2.0, requires_grad=True)   # we set requires_grad=True to let PyTorch know to keep the graph
# b = torch.tensor(1.0, requires_grad=True)
# c = a + b
# d = b + 1
# e = c * d
#
# print('c', c)
# print('d', d)
# print('e', e)



"""
Auto gradient
"""
# def f(x):
#     return (x - 2) ** 2
#
#
# def fp(x):
#     return 2 * (x - 2)
#
#
# x = torch.tensor([1.0], requires_grad=True)
#
# y = f(x)
# y.backward()        # .backward() computes all the gradients of y at once
#
# print('Analytical f\'(x):', fp(x))
# print('PyTorch\'s f\'(x):', x.grad)


# def g(w):
#     return 2 * w[0] * w[1] + w[1] * torch.cos(w[0])
#
#
# def grad_g(w):
#     return torch.tensor([2 * w[1] - w[1] * torch.sin(w[0]), 2 * w[0] + torch.cos(w[0])])
#
#
# w = torch.tensor([np.pi, 1], requires_grad=True)
#
# z = g(w)
# z.backward()
#
# print('Analytical grad g(w)', grad_g(w))
# print('PyTorch\'s grad g(w)', w.grad)


"""
Using the gradients: Gradient Descent
"""
# x = torch.tensor([5.0], requires_grad=True)
# step_size = 0.25
#
# print('iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch')
# for i in range(15):
#     y = f(x)
#     y.backward()    # computing the gradient
#     print('{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}'.format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))
#     # .item() 返回一个具体的数值
#     # .data() 返回一个tensor
#
#     x.data = x.data - step_size * x.grad    # perform a GD update step
#
#     # We need to zero the grad variable since the backward()
#     # call accumulates the gradients in .grad instead of overwriting.
#     # The detach_() is for efficiency. You do not need to worry too much about it.
#     x.grad.detach_()
#     x.grad.zero_()

"""
Linear Regression
"""
# make a simple linear dataset with some noise
# d = 2
# n = 50
# X = torch.randn(n, d)
# true_w = torch.tensor([[-1.0], [2.0]])
# y = X @ true_w + torch.randn(n, 1) * 0.1    # @: 矩阵乘法运算-点乘
# print('X shape:', X.shape)
# print('y shape:', y.shape)
# print('w shape:', true_w.shape)
#
#
# def model(X, w):
#     return X @ w
#
#
# def rss(y, y_hat):
#     return torch.norm(y - y_hat) ** 2 / n
#
#
# def grad_rss(X, y, w):
#     return -2 * X.t() @ (y - X @ w) / n
#
#
# w = torch.tensor([[1.], [0]], requires_grad=True)
# y_hat = model(X, w)
#
# loss = rss(y, y_hat)
# loss.backward()
#
# print('Analytical gradient:', grad_rss(X, y, w).detach().view(2).numpy())
# print('PyTorch\'s gradient:', w.grad.view(2).numpy())
#
# step_size = 0.1     # learning rate
# print('iter,\tloss,\tw')
# for i in range(20):
#     y_hat = model(X, w)
#     loss = rss(y, y_hat)
#
#     loss.backward()
#
#     w.data = w.data - step_size * w.grad
#     print('{}, \t{:.2f},\t{}'.format(i, loss.item(), w.view(2).detach().numpy()))
#
#     # We need to zero the grad variable since the backward()
#     # call accumulates the gradients in .grad instead of overwriting.
#     # The detach_() is for efficiency. You do not need to worry too much about it.
#     w.grad.detach()
#     w.grad.zero_()
#
# print('\ntrue w\t\t', true_w.view(2).numpy())
# print('estimated w\t', w.view(2).detach().numpy())

"""
nn Module:
    https://pytorch.org/docs/stable/nn.html
"""

"""
Linear Module
"""
# d_in = 3
# d_out = 4
# linear_module = nn.Linear(d_in, d_out)
#
# example_tensor = torch.tensor([[1.,2,3], [4,5,6]])
# # applys a linear transformation to the data
# transformed = linear_module(example_tensor)
# print('example_tensor', example_tensor.shape)
# print('transormed', transformed.shape)
# print()
# print('We can see that the weights exist in the background\n')
# print('W:', linear_module.weight)
# print('b:', linear_module.bias)

"""
Activation Functions
"""
# activation_fn = nn.ReLU() # we instantiate an instance of the ReLU module
# example_tensor = torch.tensor([-1.0, 1.0, 0.0])
# activated = activation_fn(example_tensor)
# print('example_tensor', example_tensor)
# print('activated', activated)

"""
Sequential
"""
# d_in = 3
# d_hidden = 4
# d_out = 1
#
# model = torch.nn.Sequential(
#                             nn.Linear(d_in, d_hidden),
#                             nn.Tanh(),
#                             nn.Linear(d_hidden, d_out),
#                             nn.Sigmoid()
#                             )
#
# example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
# transformed = model(example_tensor)
# print('transformed', transformed.shape)
#
# # we can access all of the parameters (of any nn.Module) with the parameters() method
# params = model.parameters()
# for param in params:
#     print(param)

"""
Loss functions
"""
mse_loss_fn = nn.MSELoss()

input = torch.tensor([[0., 0, 0]])
target = torch.tensor([[1., 0, -1]])

loss = mse_loss_fn(input, target)

print(loss)

"""
torch.optim:
    https://pytorch.org/docs/stable/optim.html
"""


