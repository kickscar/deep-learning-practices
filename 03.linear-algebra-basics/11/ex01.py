# coding: utf-8
# sigmoid graph
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


x = np.arange(-10, 10, 0.1)
y = sigmoid(x)

fig, subplots = plt.subplots(1, 1)
subplots.plot(x, y, 'k-')

plt.show()


