# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient)
import numpy as np


def softmax_func(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 대책
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def loss(w, x, t):
    z = np.dot(x, w)
    y = softmax_func(z)
    e = cross_entropy_error(y, t)
    return e


def numerical_gradient(f, x, data_l):
    h = 1e-4  # 0.0001
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
