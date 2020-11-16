# np.full() 함수

import numpy as np


# 1차원 모든요소가 5인 size 2 배열
arr1 = np.full(2, 5)
print(arr1, arr1.shape, arr1.dtype)


# 2차원 모든요소가 3인 4 x 3 배열
arr2 = np.full((4, 3), 3)
print(arr2, arr2.shape, arr2.dtype)
