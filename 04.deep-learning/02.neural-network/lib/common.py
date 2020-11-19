# coding: utf-8
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

    batch_size = y.shape[0]
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


# Numerical Gradient for Tensor 1(Vector)
def numerical_gradient_tensor1(f, x, data_l):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x, *data_l)

        x[i] = tmp - h
        h2 = f(x, *data_l)

        gradient[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

    return gradient


# Numerical Gradient for Tensor 1(Vector), 2(Matrix), ....n
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
