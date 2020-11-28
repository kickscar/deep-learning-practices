# SoftmaxWithLoss Layer Test
import sys
import os
from pathlib import Path
import numpy as np

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import SoftmaxWithLoss
    from common import softmax
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

_x = np.array([2.6, 3.9, 5.6])
_t = np.array([0, 0, 1])

layer = SoftmaxWithLoss()
loss = layer.forward(_x, _t)
dx = layer.backward()
print(loss, dx)


# =================================================================


def foward_propagation(x):
    y = softmax(x)
    return y


network.foward_propagation = foward_propagation
loss = network.loss(_x, _t)
print(loss)
