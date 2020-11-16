# coding: utf-8
# 영행렬
import numpy as np

# 직접 형상을 전달
m1 = np.zeros((3, 3))
print(m1)

# 행렬을 전달하여 같은 형상의 영행렬을 생성한다.
m2 = np.zeros_like(np.arange(1, 10).reshape(3, 3))
print(m2)



