# Backpropagation: Affine -> ReLU -> Affine -> SoftmaxWithLoss Layers Test (Batch)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Affine, ReLU, SoftmaxWithLoss
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

    idxaffine = 0
    gradient = dict()

    for layer in layers:
        if type(layer).__name__ == 'Affine':
            idxaffine += 1
            gradient[f'w{idxaffine}'] = layer.dw
            gradient[f'b{idxaffine}'] = layer.db

    return gradient


network.forward_propagation = forward_propagation
network.loss = loss


# =================================================================


# N = 2

# 1. load training/test data
_x, _t = np.array([
    [2.6, 3.9, 5.6],
    [1.76, 0.98, 4.16],
]), np.array([
    [0, 0, 1],
    [1, 0, 0]
])

# 2. hyperparamters


# =================================================================


# 3. initialize network
network.initialize(3, 2, 3)

layers = [
    Affine(network.params['w1'], network.params['b1']),
    ReLU(),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

grad = network.numerical_gradient_net(_x, _t)
print(grad, grad['w1'].shape, grad['b1'].shape, grad['w2'].shape, grad['b2'].shape)

network.params['w1'] -= 0.1 * grad['w1']
network.params['b1'] -= 0.1 * grad['b1']
network.params['w2'] -= 0.1 * grad['w2']
network.params['b2'] -= 0.1 * grad['b2']
loss = network.loss(_x, _t)
print(f'loss: {loss}')


# =================================================================


# 3. initialize network
network.initialize(3, 2, 3)

layers = [
    Affine(network.params['w1'], network.params['b1']),
    ReLU(),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

grad = backpropagation_gradient_net(_x, _t)
print(grad, grad['w1'].shape, grad['b1'].shape, grad['w2'].shape, grad['b2'].shape)

network.params['w1'] -= 0.1 * grad['w1']
network.params['b1'] -= 0.1 * grad['b1']
network.params['w2'] -= 0.1 * grad['w2']
network.params['b2'] -= 0.1 * grad['b2']
loss = network.loss(_x, _t)
print(f'loss: {loss}')
