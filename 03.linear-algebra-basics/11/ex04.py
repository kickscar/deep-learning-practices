# coding: utf-8
# Logistic Regression(수치미분)
import numpy as np
from matplotlib import pyplot as plt


# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


# loss function 함수
def loss_func(x, data_x, data_y):
    e = np.mean([-(dy * np.log(sigmoid(x[0] * dx + x[1])) + (1 - dy) * np.log(1 - sigmoid(x[0] * dx + x[1]))) for dx, dy in tuple(zip(data_x, data_y))])
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
times = [2, 4, 6, 8, 10, 12, 14]
passed = [0, 0, 0, 1, 1, 1, 1]


# 경사하강법
result = gradient_descent(loss_func, np.array([0., 0.]), lr=0.5, epoch=50000, data_l=(times, passed))

# graph
fig, subplots = plt.subplots(1, 1)
subplots.scatter(times, passed)
x2 = list(np.arange(0, 15, 0.1))
subplots.plot(x2, [sigmoid(result[0] * value + result[1]) for value in x2])
plt.show()


# predict
print(sigmoid(result[0] * 7 + result[1]))
