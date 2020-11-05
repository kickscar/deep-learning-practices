# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현1: 은닉1층 전달

import numpy as np

W1 = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 1.]])
X = np.array([1., 5.])
B1 = np.array([0.1, 0.2, 0.3])

print(f'W1 Dimension:{W1.shape}')  # 3 X 2
print(f'X Dimension:{X.shape}')    # 2 X (1)
print(f'B1 Dimension:{B1.shape}')   # 3 X (1)

A1 = np.dot(W1, X) + B1
print(f'A1={A1}')
