# 편미분(Partial Differentiation): x1을 4로 고정
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_diff
except ImportError:
    print('Library Module Can Not Found')


def f(x0):
    return x0**2 + 4.**2


# (x0, x1) = (3, 4)
print(f'Numerical Partial Differentiation Value:{numerical_diff(f, np.array(3.))}')
