# coding: utf-8
# 일차방정식
from matplotlib import pyplot as plt
import numpy as np


def f1(x):
    return 6*x + 20


data_x = np.arange(-10, 10, 0.1)
data_y = f1(data_x)

fig, subplots = plt.subplots()

plt.axvline(x=0, color='r')     # draw x = 0 axes
plt.axhline(y=0, color='r')     # draw y = 0 axes

subplots.plot(data_x, data_y)

subplots.set_xticks([-10, -5, 0, 5, 10])
subplots.set_yticks([-100, -50, 0, 50, 100])

plt.show()
