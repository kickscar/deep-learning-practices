# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현5: 출력층 전달
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex04 import z2
except ImportError:
    raise ImportError("Library Module Can Not Found")

print('\n= 신호전달 구현5: 출력층 전달 ===============================')

print(f'z2 dimension: {z2.shape}')  # 2 vector

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
print(f'w3 dimension: {w3.shape}')  # 2 X 2 matrix

b3 = np.array([0.1, 0.2])
print(f'b3 dimension: {b3.shape}')  # 2 vector

a3 = np.dot(w3, z2) + b3
print(f'a3 = {a3}')
