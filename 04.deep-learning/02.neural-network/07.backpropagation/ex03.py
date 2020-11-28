# SoftmaxWithLoss Layer Test
import sys
import os
from pathlib import Path
import numpy as np

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import SoftmaxWithLoss
except ImportError:
    raise ImportError("Library Module Can Not Found")

x = np.array([2.6, 3.9, 5.6])
t = np.array([0, 0, 1])

layer = SoftmaxWithLoss()
loss = layer.forward(x, t)
dx = layer.backward()
print(loss, dx)


#
# [ 0.01346538  0.04940849 -0.06287387]
#
