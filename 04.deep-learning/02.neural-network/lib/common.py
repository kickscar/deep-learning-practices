# coding: utf-8
from inspect import signature

import numpy as np


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


# relu activation function
def relu(x):
    # return x if x > 0 else 0
    return np.maximum(0, x)


# identity activation function
def identity(x):
    return x


# softmax activation function
def softmax_oveflow(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 대책
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


# sum squares error(SSE) function
def sum_squares_error(y, t):
    e = 0.5 * np.sum((y - t) ** 2)
    return e


# cross entropy error(loss) function
# support only for one-hot & non-batch
def cross_entropy_error_non_batch(y, t):
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta))


# sum squares error(SSE) function
# support only for one-hot & batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    batch_sz = y.shape[0]
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta)) / batch_sz


# Numerical Differentiation
# x, t를 파라미터로 받는 기울기 함수
def numerical_diff1(f, x, data_in, data_out):
    h = 1e-4
    gradient = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        h1 = f(x, data_in, data_out)            # f(x+h)

        x[idx] = tmp_val - h
        h2 = f(x, data_in, data_out)            # f(x-h)
        gradient[idx] = (h1 - h2) / (2 * h)

        x[idx] = tmp_val                        # 값복원
        it.iternext()

    return gradient


# Gradient
numerical_gradient1 = numerical_diff1


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
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad