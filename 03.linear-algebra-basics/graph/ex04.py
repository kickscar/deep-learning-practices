# coding: utf-8
# 로지스틱회귀 #1
from matplotlib import pyplot as plt
import numpy as np


def f1(x):
    return np.where(x >= 8, 1, 0)


data_x = np.array([2, 4, 6, 8, 10, 12, 14])
data_y = f1(data_x)


fig, subplots = plt.subplots()

subplots.scatter(data_x, data_y)


plt.show()
