
from matplotlib import pyplot as plt
import numpy as np


def f(x):
    return 2 ** x


data_x = np.arange(-2, 5, 0.1)
data_y = f(data_x)

fig, subplots = plt.subplots()

plt.axvline(x=0, color='r')     # draw x = 0 axes
plt.axhline(y=0, color='r')     # draw y = 0 axes

subplots.plot(data_x, data_y)

subplots.set_xticks([])
subplots.set_yticks([])

plt.show()
