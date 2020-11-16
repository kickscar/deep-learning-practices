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

data_x = np.arange(-10, 10, 0.1)
data_y = sigmoid(data_x)

# fig, subplots = plt.subplots(1, 1)
# subplots.plot(data_x, data_y, 'k-')
plt.plot(data_x, data_y)
# plt.ylim(-0.1, 1, 1)

plt.show()
