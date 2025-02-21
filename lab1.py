import torch
import torch.nn as nn

import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt

# integer = torch.tensor(1234)
# decimal = torch.tensor(3.14159265359)
#
# print(f"`integer` is a {integer.ndim}-d Tensor: {integer}")
# print(f"`decimal` is a {decimal.ndim}-d Tensor: {decimal}")

# fibonacci = torch.tensor([1,1,2,3,5,8])
# count_to_100 = torch.tensor(range(100))
#
# print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
# print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")

# 2d matrices and higher-rank tensors
# matrix = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
# assert isinstance(matrix, torch.Tensor), "matrix must be a torch Tensor object"
# assert matrix.ndim == 2
#
# images = torch.zeros(10, 3, 256, 256)
# assert isinstance(images, torch.Tensor), "images must be a torch Tensor object"
# assert images.ndim == 4, "images must have 4 dimensions"
# assert images.shape == (10, 3, 256, 256), "images is incorrect shape"
# print(f"images is a {images.ndim}-d Tensor with shape: {images.shape}")

# computation on tensors

# a = torch.tensor(15)
# b = torch.tensor(61)
# c1 = torch.add(a, b)
# c2 = a + b
# print(f"c1: {c1}")
# print(f"c2: {c2}")

# def func(a, b):
#     c = torch.add(a,b)
#     d = torch.subtract(b,1)
#     e = torch.multiply(c, d)
#     return e
#
# a, b = 1.5, 2.5
# e_out = func(a,b)
# print(f"e_out: {e_out}")

# neural networks in pytorch

# class DenseLayer(torch.nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(DenseLayer, self).__init__()
#         self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
#         self.bias = torch.nn.Parameter(torch.randn(num_outputs))
#
#     def forward(self, x):
#         z = torch.matmul(x, self.W) + self.bias
#         y = torch.sigmoid(z)
#         return y
#
# num_inputs = 2
# num_outputs = 3
# layer = DenseLayer(num_inputs, num_outputs)
# x_input = torch.tensor([[1, 2.]])
# y = layer(x_input)
#
# print(f"input shape: {x_input.shape}")
# print(f"output shape: {y.shape}")
# print(f"output result: {y}")

# class LinearWithSigmoidActivation(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearWithSigmoidActivation, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, inputs):
#         linear_ouput = self.linear(inputs)
#         output = self.activation(linear_ouput)
#         return output
#
# n_input_nodes = 2
# n_output_nodes = 3
# model = LinearWithSigmoidActivation(n_input_nodes, n_output_nodes)
# x_input = torch.tensor([[1, 2.]])
# y = model(x_input)
# print(f"input shape: {x_input.shape}")
# print(f"output shape: {y.shape}")
# print(f"output result: {y}")

# nn.Module allows us to define custom models

# class LinearButSometimesIdentity(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearButSometimesIdentity, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#
#     def forward(self, inputs, isidentity=False):
#         if isidentity:
#             return inputs
#         else:
#             return self.linear(inputs)
#
# model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
# x_input = torch.tensor([1,2.])
#
# out_with_linear = model(x_input)
# out_with_identity = model(x_input, isidentity=True)
# print(f"input: {x_input}")
# print("Network linear output: {}; network identity output: {}".format(out_with_linear, out_with_identity))

# automatic differentiation
## gradient computation y = x^2

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()

dy_dx = x.grad
print("dy_dx of y=x^2 at x=3.0 is: ", dy_dx)
assert dy_dx == 6.0


## function minization with autograd and gradient descent

x = torch.randn(1)
print(f"initializing x = {x.item()}")
learning_rate = 1e-2
history = []
x_f = 4 # target val

# run gradient descent for num iter, compute loss, and gradient of loss wrt x, perform update
for i in range(500):
    x = torch.tensor([x], requires_grad=True)
    loss = (x - x_f) ** 2
    loss.backward()
    x = x.item() - learning_rate * x.grad

    history.append(x.item())


plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()