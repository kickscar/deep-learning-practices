# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현6: 출력층 활성함수 𝜎() 적용
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity
    from ex05 import a3
except ImportError:
    raise ImportError("Library Module Can Not Found")

print('\n= 신호전달 구현6: 출력층 활성함수 𝜎() 적용 ====================')

print(f'a3 dimension: {a3.shape}')  # 2 vector
y = identity(a3)
print(f'y = {y}')
