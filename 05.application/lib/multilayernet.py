# coding: utf-8
# Multi-Layer Neural Network(Based on Backpropagation Graph Layers)
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


def initialize(input_size, hidden_sizes, output_size, init_weight=0.01, init_params=None):
    hidden_count = len(hidden_sizes)

    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_sizes[0])
        params['b1'] = np.zeros(hidden_sizes[0])
        params[f'w{hidden_count+1}'] = init_weight * np.random.randn(hidden_sizes[hidden_count-1], output_size)
        params[f'b{hidden_count+1}'] = np.zeros(output_size)
        for idx in range(1, hidden_count):
            params[f'w{idx+1}'] = init_weight * np.random.randn(hidden_sizes[idx-1], hidden_sizes[idx])
            params[f'b{idx+1}'] = np.zeros(hidden_sizes[idx])
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


def predict(x):
    y = forward_propagation(x)
    y = np.argmax(y, axis=1)

    return np.int(y)


def accuracy(x, t):
    y = forward_propagation(x)

    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def loss(x, t):
    y = forward_propagation(x, t)
    return y


def backpropagation_gradient(x, t):
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


def numerical_gradient(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            temp = param[idx]

            param[idx] = float(temp) + h
            h1 = loss(x, t)

            param[idx] = float(temp) - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = temp   # 값복원
            it.iternext()

        gradient[key] = param_gradient

    return gradient
