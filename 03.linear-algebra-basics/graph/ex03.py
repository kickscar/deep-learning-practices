# 이차방정식 #2

from matplotlib import pyplot as plt
import numpy as np


def f1(x):
    return 20*x**2


def f2(x):
    return 20*(x-2)**2 + 500


data_x = np.arange(-18, 20, 0.1)
data_x2 = np.arange(-18, 20, 0.1)
data_y1 = f1(data_x)
data_y2 = f2(data_x2)

fig, subplots = plt.subplots()

plt.axvline(x=0, color='r')     # draw x = 0 axes
plt.axhline(y=0, color='r')     # draw y = 0 axes

subplots.plot(data_x, data_y1, '--')
subplots.plot(data_x2, data_y2, 'r-')

subplots.set_xticks([])
subplots.set_yticks([])

plt.show()

