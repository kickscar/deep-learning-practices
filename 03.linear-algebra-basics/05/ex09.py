# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt


def f1(x):
    return -np.log(x)


def f2(x):
    return -np.log(1-x)


data_x1 = np.arange(0.01, 1.5, 0.01)
data_y1 = f1(data_x1)

data_x2 = np.arange(-0.01, 1.5, 0.01)
data_y2 = f2(data_x2)


fig, subplots = plt.subplots()

subplots.axvline(x=0, color='r')
subplots.axhline(y=0, color='r')

subplots.plot(data_x1, data_y1)
subplots.plot(data_x2, data_y2)

subplots.set_xticks([])
subplots.set_yticks([])

plt.show()
