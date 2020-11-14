# coding: utf-8
# ReLU graph
import numpy as np
from matplotlib import pyplot as plt


def relu(x):
    # return x if x > 0 else 0
    return np.maximum(0, x)


data_x = np.arange(-10, 10, 0.1)
data_y = relu(data_x)

plt.plot(data_x, data_y)
plt.show()
