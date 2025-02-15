# forward pass
x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# store parameters and gradients in a dict
parameters = {
    'w': w,
    'b': b
}

gradients = {
    'w': [0.0] * len(w),
    'b': 0.0,
    'x': [0.0] * len(x)
}

# mult input and weight
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# add bias, output from neuron
z = xw0 + xw1 + xw2 + b

# backward pass
y = max(z, 0)

# derivative from next layer
dvalue = 1.0

# derivative of relu and chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)

gradients['w'] = [drelu_dz * xi for xi in x]
gradients['b'] = drelu_dz * 1
gradients['x'] = [drelu_dz * wi for wi in w]

print("gradients:")
print(f"dw: {gradients['w']}")
print(f"db: {gradients['b']}")
print(f"dx: {gradients['x']}")




