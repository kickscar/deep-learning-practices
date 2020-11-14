# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현3: 은닉2층 전달
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
print('===============================')

W2 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B2 = np.array([0.1, 0.2])

print(f'W2 Dimension:{W2.shape}')  # 2 X 3
print(f'Z1 Dimension:{Z1.shape}')  # 3 Vector
print(f'B2 Dimension:{B2.shape}')  # 2 Vector

A2 = np.dot(W2, Z1) + B2
print(f'A2={A2}')
