# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현2: 은닉1층 활성함수 h() 적용
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex01 import a1
except ImportError:
    raise ImportError("Library Module Can Not Found")

print('\n= 신호전달 구현2: 은닉1층 활성함수 h() 적용 ===================')

print(f'a1 dimension: {a1.shape}')  # 3 vector
z1 = sigmoid(a1)
print(f'z1 = {z1}')
