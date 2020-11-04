# 항등행렬

import numpy as np

m = np.identity(3)
print(m)

s = np.arange(1, 10).reshape(3, 3)
print(s)

print(np.dot(s, m))



