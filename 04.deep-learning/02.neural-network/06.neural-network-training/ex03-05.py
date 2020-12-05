# coding: utf-8
# 신경망 기울기(Neural Network Gradient): Parameters(가중치 w, 편향 b) 편미분 과정 #5
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")

x = np.array([0.6, 0.9])                    # input (x)         2 vector
t = np.array([0, 0, 1])                     # label (one-hot)   3 vector
params = {
    'w1': np.random.randn(2, 3),            # weight            2 x 3 matrix
    'b1': np.array([0.45, 0.23, 0.11])      # bias              3 vector
}


def foward_propagation():
    w1 = params['w1']
    b1 = params['b1']

    z = np.dot(x, w1) + b1
    y = softmax(z)

    return y


def loss():
    y = foward_propagation()
    e = cross_entropy_error(y, t)

    return e


def numerical_gradient_net():
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_grad = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = param[idx]

            # f(x+h)
            param[idx] = float(tmp_val) + h
            h1 = loss()

            # f(x-h)
            param[idx] = tmp_val - h
            h2 = loss()

            # 기울기
            param_grad[idx] = (h1 - h2) / (2 * h)

            # 값복원
            param[idx] = tmp_val

            it.iternext()

        gradient[key] = param_grad

    return gradient


# test
g = numerical_gradient_net()
print(g)
