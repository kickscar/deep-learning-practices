# coding: utf-8
# 단위행렬
# 대각행렬 중에 주 대각선의 모든 요소가 1인 행렬
import numpy as np

m = np.eye(3, 3)
print(m)

# 기본적으로 단위행렬은 대각행렬이기 때문에 대각선 위치 번호에 따라 대각행렬과 같은 모습으로 변환된다.
m = np.eye(3, 3, k=1)
print(m)

m = np.eye(3, 3, k=-1)
print(m)

