# 수치미분(Numerical Differentiation) VS 해석미분(Analytic Differentiation)
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
    return 20*(x-2)**2+500


def analytic_diff(x):
    return 40 * x - 80


print(f'Numerical Differentiation Value:{numerical_diff(f, np.array(5.))}')
print(f'Analytic Differentiation Value:{analytic_diff(np.array(5.))}')
