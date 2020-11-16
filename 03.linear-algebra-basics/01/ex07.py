# coding: utf-8
# 대각행렬

import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# diag() 함수에 행렬을 파라미터로 전달하면 주 대각선 요소를 벡터로 반환한다.
v = np.diag(m)
print(v)


