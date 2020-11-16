# coding: utf-8
# Matrix 확인
import numpy as np

"""

1. 행렬은 두 개의 축 2차원(Tensor 2)이다.
2. 수평 방향을 행(row), 수직방향을 열(column)  

"""
m = np.array([[50], [100]])
print(m)
print(m.ndim, m.shape)
