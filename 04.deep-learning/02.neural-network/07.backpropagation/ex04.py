# Affine + SoftmaxWithLoss Layer Test
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Affine, SoftmaxWithLoss
    from common import softmax
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


def forward_propagation(x, t=None):
    for layer in layers:
        x = layer.forward(x, t) if t is not None and type(layer).__name__ == 'SoftmaxWithLoss' else layer.forward(x)

    return x


def backward_propagation(dout):
    for layer in layers[::-1]:
        dout = layer.backward(dout)

    return dout


def loss(x, t):
    y = forward_propagation(x, t)
    return y


def backpropagation_gradient_net(x, t):
    forward_propagation(x, t)
    backward_propagation(1)
    for layer in layers:
        if type(layer).__name__ == 'Affine':
            print(layer.dw)
            print(layer.db)


network.foward_propagation = forward_propagation
network.loss = loss


# 1. load training/test data
_x, _t = np.array([2.6, 3.9, 5.6]), np.array([0, 0, 1])

# 2. hyperparamters

# 3. initialize network
network.initialize(3, 2, 3)

layers = [
    Affine(network.params['w1'], network.params['b1']),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

gradient = network.numerical_gradient_net(_x, _t)
print(gradient)


# =================================================================

# 1. load training/test data
_x, _t = np.array([[2.6, 3.9, 5.6]]), np.array([0, 0, 1])

# 2. hyperparamters

# 3. initialize network
network.initialize(3, 2, 3)

layers = [
    Affine(network.params['w1'], network.params['b1']),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

backpropagation_gradient_net(_x, _t)
