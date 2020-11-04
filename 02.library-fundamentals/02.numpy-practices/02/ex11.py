# 배열 차원 변환

import numpy as np

arr1 = np.arange(1, 13)
print(arr1.shape)

arr2 = arr1.reshape(2, 6)
print(arr2)

arr3 = arr1.reshape(2, 2, 3)
print(arr3)



