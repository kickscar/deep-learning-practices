# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


def loss(w, x, t):
    z = np.dot(x, w)
    y = softmax(z)
    e = cross_entropy_error(y, t)
    return e


def numerical_gradient(f, x, data_l):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x, *data_l)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x, *data_l)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


# def numerical_gradient(f, x, data_l):
#     h = 1e-4
#     gradient = np.zeros_like(x)
#
#     for i in range(x.size):
#         tmp = x[i]
#
#         x[i] = tmp + h
#         h1 = f(x, *data_l)
#
#         x[i] = tmp - h
#         h2 = f(x, *data_l)
#
#         gradient[i] = (h1 - h2) / (2 * h)
#         x[i] = tmp
#
#     return gradient


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])


w = np.random.randn(2, 3)


gradient = numerical_gradient(loss, w, (x, t))
print(gradient)
