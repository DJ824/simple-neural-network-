import sys
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# batches of input samples, 3 samples w/ 4 features
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# 3 neurons, each taking 4 inputs, calculates outputs for all samples thru first layer of 3 neurons
# we have to transpose the weights in order to match for matrix mult (m x n, n x m)
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# another 3 neurons, each taking 3 inputs (the outputs from the previous layer)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap = 'brg')
plt.show()


