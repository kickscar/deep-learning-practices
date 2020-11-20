# coding: utf-8
# sigmoid 함수의 특징1
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


x = np.arange(-10, 10, 0.1)

fig, subplots = plt.subplots(1, 1)
for a in np.arange(0, 10, 1):
    y = sigmoid(a * x)
    subplots.plot(x, y)

plt.show()
