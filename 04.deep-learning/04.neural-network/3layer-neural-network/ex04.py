# 3층 신경망 구현하기 – 신호전달 구현4: 은닉2층 활성함수 h() 적용

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


W1 = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 1.]])
X = np.array([1., 5.])
B1 = np.array([0.1, 0.2, 0.3])

print(f'W1 Dimension:{W1.shape}')  # 3 X 2
print(f'X Dimension:{X.shape}')    # 2 X (1)
print(f'B1 Dimension:{B1.shape}')  # 3 X (1)

A1 = np.dot(W1, X) + B1
print(f'A1={A1}')
print('===============================')

Z1 = sigmoid(A1)
print(f'Z1={Z1}')
print('===============================')

W2 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B2 = np.array([0.1, 0.2])

print(f'W2 Dimension:{W2.shape}')  # 2 X 3
print(f'Z1 Dimension:{Z1.shape}')  # 3 X (1)
print(f'B2 Dimension:{B2.shape}')  # 2 X (1)

A2 = np.dot(W2, Z1) + B2
print(f'A2={A2}')
print('===============================')

Z2 = sigmoid(A2)
print(f'Z2={Z2}')

