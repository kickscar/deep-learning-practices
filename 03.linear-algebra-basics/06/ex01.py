# coding: utf-8
# 수치미분(Numerical Differentiation)
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_diff
except ImportError:
    print('Library Module Can Not Found')


# 함수 y = 20(x-2)^2 + 500
def f(x):
    return 20*(x-2)**2+500


print(f'Differentiation Value:{numerical_diff(f, np.array(2.))}')
print(f'Differentiation Value:{numerical_diff(f, np.array(1.9))}')
