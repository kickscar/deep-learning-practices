# 기울기(Gradient)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import gradient_descent
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


def function(x):
    return np.sum(x**2, axis=0)


gradient_descent(function, np.array([-3., 4.]), 0.1)
# gradient_descent(function, np.array([-3., 4.]), 10)
# gradient_descent(function, np.array([-3., 4.]), 1e-10)

