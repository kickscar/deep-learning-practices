# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt


def f1(x):
    return 2 ** x


def f2(x):
    return np.log2(x)


data_x1 = np.arange(-2, 5, 0.1)
data_y1 = f1(data_x1)

data_x2 = data_y1
data_y2 = f2(data_x2)

fig, subplots = plt.subplots(2, 1)

subplots[0].axvline(x=0, color='r')
subplots[0].axhline(y=0, color='r')
subplots[0].plot(data_x1, data_y1)
subplots[0].set_xticks([])
subplots[0].set_yticks([])

subplots[1].axvline(x=0, color='r')
subplots[1].axhline(y=0, color='r')
subplots[1].plot(data_x2, data_y2)
subplots[1].set_xticks([])
subplots[1].set_yticks([])


plt.show()
