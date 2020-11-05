# coding: utf-8
# 전치행렬(transpose matrix)
import numpy as np

# a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print('=====================')


a1 = a.T
print(a1)
print(a1.shape)
print('=====================')


a2 = np.transpose(a)
print(a2)
print(a2.shape)
print('=====================')


a3 = np.swapaxes(a, 0, 1)
print(a3)
print(a3.shape)
print('=====================')