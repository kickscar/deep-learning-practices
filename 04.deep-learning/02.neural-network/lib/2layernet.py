# coding: utf-8
# 2 Layer Neural Network
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


_params = dict()


def init_net(sz_input, sz_hidden, sz_output, w_init=0.01):
    _params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)
    _params['b1'] = np.zeros(sz_hidden)
    _params['w2'] = w_init * np.random.randn(sz_hidden, sz_output)
    _params['b2'] = np.zeros(sz_output)


def _foward_propagation(x):
    w1 = _params['w1']
    b1 = _params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = sigmoid(a1)

    w2 = _params['w2']
    b2 = _params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)

    return y


def _loss(x, t):
    y = _foward_propagation(x)
    e = cross_entropy_error(y, t)

    return e


def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in _params:
        param = _params[key]
        param_grad = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = param[idx]

            # f(x+h)
            param[idx] = float(tmp_val) + h
            h1 = _loss(x, t)

            # f(x-h)
            param[idx] = tmp_val - h
            h2 = _loss(x, t)

            # 기울기
            param_grad[idx] = (h1 - h2) / (2 * h)

            # 값복원
            param[idx] = tmp_val

            it.iternext()

        gradient[key] = param_grad

    return gradient
