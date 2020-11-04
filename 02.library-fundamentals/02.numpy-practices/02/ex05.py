import numpy as np

arr1 = np.arange(0, 10)
print(arr1.shape)

# 수직축 추가하기
arr2 = arr1[:, np.newaxis]
print(arr2.shape)

# 수평축 추가하기
arr3 = arr1[np.newaxis, :]
print(arr3.shape)
