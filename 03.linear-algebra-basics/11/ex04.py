# coding: utf-8
# Logistic Regression(수치미분)
import os
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import gradient_descent
except ImportError:
    print('Library Module Can Not Found')


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


# loss function
def loss(x, data_in, data_out):
    e = data_out * np.log(sigmoid(x[:-1] * data_in + x[-1:])) + (1-data_out) * np.log(1-sigmoid(x[:-1] * data_in + x[-1:]))
    return -1 * np.mean(e)


# data
times = np.array([2, 4, 6, 8, 10, 12, 14])
passed = np.array([0, 0, 0, 1, 1, 1, 1])


# 경사하강법
params = gradient_descent(loss, np.array([0., 0.]), lr=0.5, epoch=50000, data_in=times, data_out=passed)

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
