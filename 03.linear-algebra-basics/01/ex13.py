# coding: utf-8
# 행렬 산술연산: 덧셈

import numpy as np

m1 = np.array([[1, 2, 3], [4, 5, 6]])
print(m1)

m2 = np.array([[10, 20, 30], [40, 50, 60]])
print(m2)

m3 = m1 + m2
print(m3)

# 연산자 + 와 같은 함수다.
m3 = np.add(m1, m2)
print(m3)



