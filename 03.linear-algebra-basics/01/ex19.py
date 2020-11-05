# coding: utf-8

import numpy as np

m1 = np.array([[1, 2], [3, 4], [5, 6]])
m2 = np.array([[10, 10], [20, 20]])


# m3 = m1 @ m2
m3 = np.dot(m1, m2)

print(m3)
