# coding: utf-8
import numpy as np

m = np.array([[10, 20], [1, 2]])

print(m)
print(np.sum(m))
print(np.sum(m, axis=0))
print(np.sum(m, axis=1))
