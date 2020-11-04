# 다차원 배열 조회

import numpy as np

arr1 = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
row, col = arr1.shape

for i in range(row):
    for j in range(col):
        print(f'{i}:{j} : {arr1[i, j]}')
