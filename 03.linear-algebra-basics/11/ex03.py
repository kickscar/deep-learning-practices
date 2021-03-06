# coding: utf-8
# sigmoid 함수의 특징2
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


x = np.arange(-10, 10, 0.1)

fig, subplots = plt.subplots(1, 1)

subplots.axvline(x=0, color='k')
subplots.axhline(y=0.5, color='k')

for b in np.arange(-5, 5, 1):
    y = sigmoid(x + b)
    subplots.plot(x, y)

plt.show()
