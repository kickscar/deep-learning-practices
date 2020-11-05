# coding: utf-8
import numpy as np

m1 = np.array([[1, 2], [3, 4], [5, 6]])
m2 = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([10, 100])

m3 = np.dot(v, m2)
m4 = np.dot(m1, v)

print(m3)
print(m4)
