# coding: utf-8
# 대각행렬
import numpy as np

v = np.arange(1, 4)

# diag() 함수에  대각행렬의 주 대각선 요소를 벡터로 전달하면 대각선 요소를 제외한 나머지 요소를 모두 0로 채운 대각행렬을 만들어 준다.
m = np.diag(v)
print(m)

# 대각선 위치 번호에 따른 행렬 변화
m = np.diag(v, k=1)
print(m)

m = np.diag(v, k=-2)
print(m)
