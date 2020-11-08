# coding: utf-8
# sigmoid graph
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


data_x = list(np.arange(-10, 10, 0.1))
data_y = [sigmoid(i) for i in data_x]

fig, subplots = plt.subplots(1, 1)
subplots.plot(data_x, data_y, 'k-')

plt.show()


