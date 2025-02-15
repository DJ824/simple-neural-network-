import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize random weights and biases to be 0
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)

# create dataset
X, y = spiral_data(samples=100, classes=3)

# create dense layer w/ 2 input features and 3 output value s
dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLu()

# perform a forward pass of our training data through this layer
dense1.forward(X)

activation1.forward(dense1.output)

print(activation1.output[:5])


