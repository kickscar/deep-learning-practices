# coding: utf-8
# 다중선형회귀(수치미분)
import numpy as np
from matplotlib import pyplot as plt


def mse(x, data_x0, data_x1, data_y):
    data_y_hat = [x[0] * dx0 + x[1] * dx1 + x[2] for dx0, dx1 in tuple(zip(data_x0, data_x1))]
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
ptimes = [0, 4, 2, 3]
scores = [81, 93, 91, 97]

# 경사하강법
result = gradient_descent(mse, np.array([0., 0., 0.]), epoch=3000, data_l=(times, ptimes, scores))


# 3D 그래프
axes = plt.axes(projection='3d')
axes.scatter(times, ptimes, scores)
axes.set_xlabel('Study Hours')
axes.set_ylabel('Private Class Hours')
axes.set_zlabel('Score')

plt.show()

# predict!!
x1_p = 2
x2_p = 2
y_p = result[0] * x1_p + result[1] * x2_p + result[2]
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을 때, 받을 수 있는 점수: {int(y_p)}점')
