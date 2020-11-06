# coding: utf-8
# 경사하강법(수치미분)
import numpy as np


def mse(x, data_x, data_y):
    data_y_hat = [x[0] * dx + x[1] for dx in data_x]
    e = np.mean([(dyh - dy) ** 2 for dyh, dy in tuple(zip(data_y_hat, data_y))])
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


def gradient_descent(f, x, lr=0.01, epoch=100, data_l=None):
    for i in range(epoch):
        gradient = numerical_gradient(f, x, data_l)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 경사하강법
result = gradient_descent(mse, np.array([0., 0.]), epoch=3000, data_l=(times, scores))

# 평균제곱오차
e = mse(result, times, scores)
print(f'오차:{e}')
