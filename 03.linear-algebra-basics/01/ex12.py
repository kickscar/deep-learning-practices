# coding: utf-8
# 전치행렬(transpose matrix)
"""

1. 어떤 행렬의 행과 열을 바꿔 생성한 행렬이다.
2. 축에 대한 변경이지 요소의 변경이 아니다.

"""
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