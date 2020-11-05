# coding: utf-8
# sigmoid test3
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


data_x = list(np.arange(-10, 10, 0.1))

fig, subplots = plt.subplots(1, 1)
for b in np.arange(-5, 5, 1):
    data_y = [sigmoid(x + b) for x in data_x]
    subplots.plot(data_x, data_y)

plt.show()
