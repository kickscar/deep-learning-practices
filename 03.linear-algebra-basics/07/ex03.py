# 편미분(Partial Diffirentiation): x0, x1에 대해 미분
import os
import sys
from pathlib import Path

import numpy as np

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_partial_diff
except ImportError:
    print('Library Module Can Not Found')


# y = x0**2 + x1**2
# f(x0, x1):
def f(x):
    return x[0]**2 + x[1]**2


# (x0, x1) = (3, 4)
print(f'Numerical Partial Diffierentiation Value:{numerical_partial_diff(f, np.array([3., 4.]))}')
