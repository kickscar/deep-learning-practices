# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return -np.log(-x)


data_x = np.arange(-2, 0, 0.01)
data_y = f(data_x)


fig, subplots = plt.subplots()

subplots.axvline(x=0, color='r')
subplots.axhline(y=0, color='r')
subplots.plot(data_x, data_y)
subplots.set_xticks([])
subplots.set_yticks([])

plt.show()
