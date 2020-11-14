# coding: utf-8
# 출력함수(출력층 활성함수) 𝜎() – 항등함수(Identity Function)
import numpy as np
from matplotlib import pyplot as plt


def identity_func(x):
    return x


data_x = np.arange(-10, 10, 0.1)
data_y = identity_func(data_x)

plt.plot(data_x, data_y)
plt.show()
