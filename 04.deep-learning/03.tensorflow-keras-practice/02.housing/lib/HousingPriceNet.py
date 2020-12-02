# coding: utf-8
# HousingPriceNetwork
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import relu, mean_squares_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


params = dict()


def initialize(input_size, hidden_size, output_size, weight_init=0.01):
    params['w1'] = weight_init * np.random.randn(input_size, hidden_size)
    params['b1'] = np.zeros(hidden_size)

    params['w2'] = weight_init * np.random.randn(hidden_size, 6)
    params['b2'] = np.zeros(6)

    params['w3'] = weight_init * np.random.randn(6, output_size)
    params['b3'] = np.zeros(output_size)


def forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = relu(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2
    z2 = relu(a2)

    w3 = params['w3']
    b3 = params['b3']
    a3 = np.dot(z2, w3) + b3

    return a3


def loss(x, t):
    y = forward_propagation(x)
    e = mean_squares_error(y, t)
    return e


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
