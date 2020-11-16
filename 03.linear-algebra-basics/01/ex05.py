# coding: utf-8
import numpy as np

m = np.array([[10, 20], [1, 2]])

"""
1. 2차원 배열, 행렬의 축 기준 연산(합)
2. 축이 두 개이기 때문에 축 지정 유무에 따라 그 결과가 다르다
"""

print(m)
print(np.sum(m))
print(np.sum(m, axis=0))
print(np.sum(m, axis=1))
