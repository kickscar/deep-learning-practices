# coding: utf-8
import numpy as np

v = np.arange(1, 11)

"""

1. 1차원 배열, 벡터의 축 기준 연산(합)
2. 축이 하나기 때문에 축 지정 유무에 상관없이 결과가 같다.   

"""

print(v)
print(np.sum(v))
print(np.sum(v, axis=0))
