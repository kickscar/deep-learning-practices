# coding: utf-8
# Logistic Regression(해석미분)
import numpy as np
from matplotlib import pyplot as plt


# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def analytic_gradient(x, d_in, d_out):
    return np.array([
        d_in * (sigmoid(x[0] * d_in + x[1]) - d_out),
        sigmoid(x[0] * d_in + x[1]) - d_out
    ])


def gradient_descent(x, lr=0.01, epoch=100, data_in=None, data_out=None):
    for i in range(epoch):
        for d_in, d_out in zip(data_in, data_out):
            gradient = analytic_gradient(x, d_in, d_out)

            # 출력
            print(f'epoch={i+1}, gradient={gradient}, x={x}')

            x -= lr * gradient

    return x


# data
times = np.array([2, 4, 6, 8, 10, 12, 14])
passed = np.array([0, 0, 0, 1, 1, 1, 1])


# 경사하강법
params = gradient_descent(np.array([0., 0.]), lr=0.1, epoch=1000, data_in=times, data_out=passed)

# predict
x_p = 7
print(f'{x_p}시간 공부했을 때 합격할 확률은 {sigmoid(params[0] * x_p + params[1])} 입니다.')

# graph
x = np.arange(0, 15, 0.1)
y = sigmoid(params[0] * x + params[1])

fig, splt = plt.subplots()
splt.scatter(times, passed)
splt.plot(x, y)

plt.show()
