# 오류!

import numpy as np

# dimension error: shapes (3,2) and (3,3) not aligned
try:
    m1 = np.array([[1, 2], [3, 4], [5, 6]])
    m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    m3 = np.dot(m1, m2)

except ValueError as ex:
    print(f'error: {ex}')

