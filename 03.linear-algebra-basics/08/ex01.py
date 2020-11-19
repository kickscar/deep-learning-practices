# 기울기(Gradient)
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_gradient
except ImportError:
    print('Library Module Can Not Found')


def function(x):
    return np.sum(x**2, axis=0)


print(numerical_gradient(function, np.array([3., 4.])))
print(numerical_gradient(function, np.array([-1., -1.5])))
print(numerical_gradient(function, np.array([-0.25, -0.25])))
