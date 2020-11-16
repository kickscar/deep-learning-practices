# coding: utf-8
# 단위행렬
import numpy as np

m = np.diag([4, 4, 4])
# 대각요소의 역수를 곱해서 생성할수 도 있다.
m = m * (1 / 4)

print(m)


