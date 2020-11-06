# coding: utf-8
# 다중선형회귀(해석미분)
import numpy as np


def analytic_gradient(x, data_l):
    gradient = np.zeros_like(x)

    n = len(data_l[0])

    data_y_hat = x[0] * data_l[0] + x[1] * data_l[1] + x[2]
    arr_error = data_l[2] - data_y_hat

    gradient[0] = -(2 / n) * np.sum(arr_error * data_l[0])
    gradient[1] = -(2 / n) * np.sum(arr_error * data_l[1])
    gradient[2] = -(2 / n) * np.sum(arr_error)

    return gradient


def gradient_descent(x, lr=0.01, epoch=100, data_l=None):
    for i in range(epoch):
        gradient = analytic_gradient(x, data_l)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x


# data
times = [2, 4, 6, 8]
ptimes = [0, 4, 2, 3]
scores = [81, 93, 91, 97]

# 경사하강법
gradient_descent(np.array([0., 0., 0.]), epoch=3000, data_l=(np.array(times), np.array(ptimes), np.array(scores)))
