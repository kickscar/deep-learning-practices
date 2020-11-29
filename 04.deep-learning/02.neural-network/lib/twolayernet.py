# coding: utf-8
# Two Layer Neural Network
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


params = dict()


def initialize(szinput, szhidden, szoutput, weight_init=0.01):
    params['w1'] = weight_init * np.random.randn(szinput, szhidden)
    params['b1'] = np.zeros(szhidden)
    params['w2'] = weight_init * np.random.randn(szhidden, szoutput)
    params['b2'] = np.zeros(szoutput)


def forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = sigmoid(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)
    return y


def accuracy(x, t):
    y = forward_propagation(x)

    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def loss(x, t):
    y = forward_propagation(x)

    e = cross_entropy_error(y, t)
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
