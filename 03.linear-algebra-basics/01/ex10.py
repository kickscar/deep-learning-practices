# coding: utf-8
# 항등행렬
# 단위행렬의 다른 이름이다.
import numpy as np

m = np.identity(3)
print(m)

s = np.arange(1, 10).reshape(3, 3)
print(s)

# 생성된 단위 행렬 m과 행렬 s를 닷연산(행렬내적)을 하여도 행렬 s는 변화가 없다.
print(np.dot(s, m))



