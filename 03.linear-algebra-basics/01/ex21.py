# coding: utf-8
import numpy as np

m1 = np.array([[1, 2], [3, 4], [5, 6]])
m2 = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([10, 100])

# 벡터와의 dot 연산과 dot 연산도 가능하다.
m3 = np.dot(v, m2)
print(m3)

# 벡터와의 dot 연산과 dot 연산도 가능하다.
m4 = np.dot(m1, v)
print(m4)
