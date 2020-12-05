# coding: utf-8
import numpy as np


# identity
def identity(x):
    y = x

    return y


# sigmoid
def sigmoid(x):
    y = 1 / (1 + np.e ** (-x))

    return y


# relu
def relu(x):
    y = np.maximum(0, x)

    return y


# softmax
def softmax(x):
    # x = np.exp(x)
    # y = x / np.sum(x)

    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 대책
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    y = np.exp(x) / np.sum(np.exp(x))

    return y


# SSE(Sum Squares Error)
def sum_squares_error(y, t):
    e = 0.5 * np.sum((y - t) ** 2)
    return e


# CEE(Cross Entropy Error)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    delta = 1.e-7
    batch_size = y.shape[0]
    return -1. * np.sum(t * np.log(y + delta)) / batch_size


# Numerical Differentiation2
def numerical_diff2(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = float(tmp_val) + h
        h1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        h2 = f(x)

        gradient[idx] = (h1 - h2) / (2 * h)

        # 값복원
        x[idx] = tmp_val

        it.iternext()

    return gradient


# Gradient2
numerical_gradient2 = numerical_diff2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    gradient = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        x[idx] = float(tmp) + h
        h1 = f(x)

        x[idx] = float(tmp) - h
        h2 = f(x)

        gradient[idx] = (h1 - h2) / (2 * h)

        x[idx] = tmp
        it.iternext()

    return gradient
