# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현2: 은닉1층 활성함수 h() 적용
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


W1 = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 1.]])
X = np.array([1., 5.])
B1 = np.array([0.1, 0.2, 0.3])

print(f'W1 Dimension:{W1.shape}')  # 3 X 2
print(f'X Dimension:{X.shape}')    # 2 Vector
print(f'B1 Dimension:{B1.shape}')  # 3 Vector

A1 = np.dot(W1, X) + B1
print(f'A1={A1}')
print('===============================')

Z1 = sigmoid(A1)
print(f'Z1={Z1}')
