import copy
import os
import urllib
import urllib.request
import numpy as np
import cv2
import pickle
import copy
import matplotlib.pyplot as plt
import tarfile
from numpy.random import default_rng


class Layer_Convolutional:
    def __init__(self, n_filters, filter_size, stride=1, padding=0):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(n_filters, filter_size)
        self.stride = stride
        self.padding = padding
        self.biases = np.zeros((n_filters, 1))

        def forward(self, inputs, training):
            self.inputs = inputs
            batch_size, input_height, input_width, input_channels = inputs.shape

            output_height = ((input_height + 2 * self.padding - self.filter_size) // self.stride) + 1
            output_width = ((input_width + 2 * self.padding - self.filter_size) // self.stride) + 1

            self.output = np.zeros((batch_size, output_height, output_width, self.n_filters))

            if self.padding > 0:
                padded_inputs = np.pad(
                    inputs,
                    ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                    mode='constant'
                )

            else:
                padded_inputs = inputs

            for i in range(0, output_height):
                for j in range(0, output_width):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size

                    input_slice = padded_inputs[:, h_start:h_end, w_start:w_end, :]
                    for f in range(self.n_filters):
                        self.output[:, i, j, f] = np.sum(
                            input_slice * self.filters[f], axis=(1, 2)
                        ) + self.biases[f]


            def backward(self, dvalues):
                self.dfilters = np.zeros_like(self.filters)
                self.dbiases = np.sum(dvalues, axis=(0,1,2))

                if self.padding > 0:
                    padded_inputs = np.pad(
                    self.inputs,
                ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)),
                    mode='constant'
                )
                else:
                    padded_inputs = self.inputs

                padded_dinputs = np.zeros_like(padded_inputs)

                for i in range(dvalues.shape[0]):
                    for f in range(self.n_filters):
                        for h in range(dvalues.shape[1]):
                            for w in range(dvalues.shape[2]):

                                input_patch = self.inputs[i,
                                              h:h+self.filter_size,
                                              w:w+self.filter_size]
                                self.dfilters[f] += input_patch * dvalues[i, h, w, f]


                                filter = self.filters[f]
                                self.dinputs[i,
                                h:h+self.filter_size,
                                w:w+self.filter_size] += filter * dvalues[i, h, w, f]

                self.dbiases = np.sum(dvalues, axis=(0, 1, 2))


class Layer_MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs, training):
        self.inputs = inputs
        batch_size, input_height, input_width, channels = inputs.shape

        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, output_height, output_width, channels))
        # store max val positions for backprop
        self.cache = {}

        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(channels):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        window = inputs[b, h_start::h_end, w_start::w_end, c]

                        self.output[b, h, w, c] = np.max(window)
                        self.cache[(b, h, w, c)] = np.unravel_index(np.argmax(window), window.shape)

    def backward(self, dvalues):
        # distribute gradients to max value position
        for b in range(dvalues.shape[0]):
            for h in range(dvalues.shape[1]):
                for w in range(dvalues.shape[2]):
                    for c in range(dvalues.shape[3]):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        max_h, max_w = self.cache[(b, h, w, c)]

                        self.dinputs[b, h_start + max_h, w_start + max_w, c] = dvalues[b, h, w, c]


