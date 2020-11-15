# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient)
import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def loss(w, x, t):
    y = np.dot(x, w)
    e = cross_entropy_error(y, t)
    return e


def numerical_gradient(f, x, data_l):
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


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])


w = np.random.randn(2, 3)


gradient = numerical_gradient(loss, w, (x, t))
print(gradient)
