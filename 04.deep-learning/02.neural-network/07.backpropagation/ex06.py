# Backpropagation: ReLU Layer Test

import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import ReLU
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# test1 (vector)
layer = ReLU()

_x = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
print(_x)

_y = layer.forward(_x)
print(_y)
print(layer.mask)   # check mask


_dout = np.array([-0.1, -0.2, -0.3, 0.4, -0.5])
_dout = layer.backward(_dout)
print(_dout)


print('# =============================================')


# test2 (matrix) - 형상 주의!
_x = np.array([
    [0.1, -0.6, 1.1],
    [0.2, -0.7, 1.2],
    [0.3, -0.8, 1.3],
    [0.4, -0.9, 1.4],
])
print(_x)

_y = layer.forward(_x)
print(_y)
print(layer.mask)   # check mask

_dout = np.array([
    [-1.1, 10.6, -2.1],
    [-1.2, 20.7, -2.2],
    [-1.3, 30.8, -2.3],
    [-1.4, 40.9, -2.4],
])
_dout = layer.backward(_dout)
print(_dout)



