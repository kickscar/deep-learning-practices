# coding: utf-8
# Logistic Regression(해석미분)
import numpy as np
from matplotlib import pyplot as plt


# sigmoid 함수
def sigmoid(v):
    return 1 / (1 + np.e ** (-v))


def analytic_gradient(x, dx, dy):
    gradient = [0., 0.]

    gradient[0] = dx * (sigmoid(x[0] * dx + x[1]) - dy)
    gradient[1] = sigmoid(x[0] * dx + x[1]) - dy

    return gradient


def gradient_descent(x, lr=0.01, epoch=100, data_l=None):
    for i in range(epoch):
        for dx, dy in tuple(zip(data_l[0], data_l[1])):
            gradient = analytic_gradient(x, dx, dy)

            # 출력
            print(f'epoch={i+1}, gradient={gradient}, x={x}')

            x[0] -= lr * gradient[0]
            x[1] -= lr * gradient[1]

    return x


# data
times = [2, 4, 6, 8, 10, 12, 14]
passed = [0, 0, 0, 1, 1, 1, 1]


# 경사하강법
result = gradient_descent([0., 0.], lr=0.1, epoch=30000, data_l=(times, passed))

# graph
fig, subplots = plt.subplots(1, 1)
subplots.scatter(times, passed)
x2 = list(np.arange(0, 15, 0.1))
subplots.plot(x2, [sigmoid(result[0] * value + result[1]) for value in x2])
plt.show()


# predict
print(sigmoid(result[0] * 7 + result[1]))
