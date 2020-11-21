# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현3: 은닉2층 전달
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from ex02 import z1
except ImportError:
    raise ImportError("Library Module Can Not Found")

print('\n= 신호전달 구현3: 은닉2층 전달 ==============================')

print(f'z1 dimension: {z1.shape}')  # 3 vector

w2 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
print(f'w2 dimension: {w2.shape}')  # 2 X 3 matrix

b2 = np.array([0.1, 0.2])
print(f'b2 dimension: {b2.shape}')  # 2 vector

a2 = np.dot(w2, z1) + b2
print(f'a2 = {a2}')
