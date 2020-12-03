# coding: utf-8
# Two Layer Neural Network2(Based on Backpropagation Graph Layers)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error
    from layers import Affine, ReLU, SoftmaxWithLoss
except ImportError:
    raise ImportError("Library Module Can Not Found")


layers = []
params = dict()


def initialize(input_size, hidden_size, output_size, init_weight=0.01, init_params=None):
    hidden_count = len(hidden_size)

    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_size[0])
        params['b1'] = np.zeros(hidden_size[0])

        for idx in range(1, hidden_count):
            params[f'w{idx+1}'] = init_weight * np.random.randn(hidden_size[idx-1], hidden_size[idx])
            params[f'b{idx+1}'] = np.zeros(hidden_size[idx])

        params[f'w{hidden_count+1}'] = init_weight * np.random.randn(hidden_size[hidden_count-1], output_size)
        params[f'b{hidden_count+1}'] = np.zeros(output_size)
    else:
        globals()['params'] = init_params

    layers.append(Affine(params['w1'], params['b1']))
    layers.append(ReLU())
    for idx in range(1, hidden_count):
        layers.append(Affine(params[f'w{idx + 1}'], params[f'b{idx + 1}']))
        layers.append(ReLU())
    layers.append(Affine(params[f'w{hidden_count+1}'], params[f'b{hidden_count+1}']))
    layers.append(SoftmaxWithLoss())


def forward_propagation(x, t=None):
    for layer in layers:
        x = layer.forward(x, t) if t is not None and type(layer).__name__ == 'SoftmaxWithLoss' else layer.forward(x)

    return x


def backward_propagation(dout):
    for layer in layers[::-1]:
        dout = layer.backward(dout)

    return dout


def accuracy(x, t):
    y = forward_propagation(x)

    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def loss(x, t):
    y = forward_propagation(x, t)
    return y


def backpropagation_gradient_net(x, t):
    forward_propagation(x, t)
    backward_propagation(1)

    affine_idx = 0
    gradient = dict()

    for layer in layers:
        if type(layer).__name__ == 'Affine':
            affine_idx += 1
            gradient[f'w{affine_idx}'] = layer.dw
            gradient[f'b{affine_idx}'] = layer.db

    return gradient
