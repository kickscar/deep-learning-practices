# coding: utf-8
# 이차방정식
import numpy as np
from matplotlib import pyplot as plt


def f1(x):
    return 5*x**2


data_x = np.arange(-10, 10, 0.1)
data_y1 = f1(data_x)

fig, subplots = plt.subplots()

plt.axvline(x=0, color='r')     # draw x = 0 axes
plt.axhline(y=0, color='r')     # draw y = 0 axes

subplots.plot(data_x, data_y1)

subplots.set_xticks([])
subplots.set_yticks([])

plt.show()

