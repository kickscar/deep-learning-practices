# 편미분(Partial Differentiation): x0, x1에 대해 미분
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_diff
except ImportError:
    print('Library Module Can Not Found')


def f(x):
    return x[0]**2 + x[1]**2


# (x0, x1) = (3, 4)
print(f'Numerical Differentiation Value:{numerical_diff(f, np.array([3., 4.]))}')
