# coding: utf-8
# sigmoid function & graph
import os
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
except ImportError:
    raise ImportError("Library Module Can Not Found")

x = np.arange(-10, 10, 0.1)
y = sigmoid(x)

# fig, subplots = plt.subplots(1, 1)
# subplots.plot(x, y, 'k-')
plt.plot(x, y)
# plt.ylim(-0.1, 1, 1)

plt.show()
