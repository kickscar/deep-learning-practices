# coding: utf-8
# Two Layer Neural Network2(Based on Backpropagation Graph Layers)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Affine, ReLU, SoftmaxWithLoss
except ImportError:
    raise ImportError("Library Module Can Not Found")


layers = []
params = dict()


def initialize(input_size, hidden_size, output_size, init_weight=0.01, init_params=None):
    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_size)
        params['b1'] = np.zeros(hidden_size)
        params['w2'] = init_weight * np.random.randn(hidden_size, output_size)
        params['b2'] = np.zeros(output_size)
    else:
        globals()['params'] = init_params

    layers.append(Affine(params['w1'], params['b1']))
    layers.append(ReLU())
    layers.append(Affine(params['w2'], params['b2']))
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


def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_grad = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp = param[idx]

            param[idx] = float(tmp) + h
            h1 = loss(x, t)

            param[idx] = float(tmp) - h
            h2 = loss(x, t)

            # 기울기
            param_grad[idx] = (h1 - h2) / (2 * h)

            # 값복원
            param[idx] = tmp

            it.iternext()

        gradient[key] = param_grad

    return gradient