import numpy as np

# passed in input gradient from the next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# 3 sets of weights, one set for each neuron, 4 inputs so 4 weights, transpose in order to mult with inputs
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T


inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])


biases = np.array([[2, 3, 0.5]])

# forward pass, dense layer & relu activation
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# backpropagation, zero out gradients that received 0 input from relu
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases


print(weights)
print(biases)