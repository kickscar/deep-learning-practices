# 행렬 산술연산 - 뺄셈

import numpy as np

m1 = np.array([[1, 2, 3], [4, 5, 6]])
m2 = np.array([[10, 20, 30], [40, 50, 60]])

# m3 = m1 - m2
m3 = np.subtract(m1, m2)

print(m1)
print(m2)
print(m3)


