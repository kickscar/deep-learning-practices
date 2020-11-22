# coding: utf-8
# 출력함수(출력층 활성함수) 𝜎() – 항등함수(Identity Function)
import os
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity
except ImportError:
    raise ImportError("Library Module Can Not Found")


x = np.arange(-10, 10, 0.1)
y = identity(x)

plt.plot(x, y)
plt.show()
