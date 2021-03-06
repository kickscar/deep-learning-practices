# coding: utf-8
# Two Layer Neural Network
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import common
except ImportError:
    raise ImportError("Library Module Can Not Found")


params = dict()


def initialize(input_size, hidden_size, output_size, init_weight=0.01, init_params=None):
    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_size)
        params['b1'] = np.zeros(hidden_size)
        params['w2'] = init_weight * np.random.randn(hidden_size, output_size)
        params['b2'] = np.zeros(output_size)
    else:
        globals()['params'] = init_params


def accuracy(x, t):
    y = _forward_propagation(x)

    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    acc = np.sum(y == t) / float(x.shape[0])

    return acc


def loss(x, t):
    return _loss(None, x, t)


def numerical_gradient(x, t):
    _loss.x, _loss.t = x, t

    gradient = dict()

    for key in params:
        param = params[key]
        param_gradienat = common.numerical_gradient(_loss, param)
        gradient[key] = param_gradienat

    return gradient


def _loss(dummy, x=None, t=None):
    if x is None:
        x = _loss.x
    if t is None:
        t = _loss.t

    y = _forward_propagation(x)
    e = common.cross_entropy_error(y, t)

    return e


def _forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = common.sigmoid(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = common.softmax(a2)

    return y
