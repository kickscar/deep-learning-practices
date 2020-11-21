# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현4: 은닉2층 활성함수 h() 적용
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex03 import a2
except ImportError:
    raise ImportError("Library Module Can Not Found")

print('\n= 신호전달 구현4: 은닉2층 활성함수 h() 적용 ===================')

print(f'a2 dimension: {a2.shape}')  # 2 vector
z2 = sigmoid(a2)
print(f'z2 = {z2}')
