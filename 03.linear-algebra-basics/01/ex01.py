# coding: utf-8
# Scalar 확인
import numpy as np

"""

1. 스칼라는 차원을 확인하면 0(Tensor 0) 이다.
2. 형상을 확인하면 빈 튜플이 반환된다.
 
"""
s = np.array(50)
print(s.ndim, s.shape)
