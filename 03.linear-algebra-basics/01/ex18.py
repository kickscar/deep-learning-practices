# coding: utf-8
import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a.shape)

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(b.shape)

c = np.dot(a, b)
print(c)

# 오류!
# dimension error: shapes (3,2) and (3,3) not aligned
# try:
#     a = np.array([[1, 2], [4, 5], [6, 7]])
#     print(a.shape)
#
#     b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     print(b.shape)
#
#     c = np.dot(a, b)
#     print(c)
# except ValueError as ex:
#     print(f'error: {ex}')





