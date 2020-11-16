# coding: utf-8
# 출력함수(출력층 활성함수) 𝜎() – 소프트맥스함수(Softmax Function)
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, softmax_oveflow
except ImportError:
    raise ImportError("Library Module Can Not Found")


# test1
a = np.array([0.3, 1., 0.78])
o = softmax(a)
print(o)

# test2: 큰값(800.)
# a = np.array([0.3, 800., 0.78])
# o = softmax_oveflow(a)
# print(o)

# test3: 큰값(800.)
a = np.array([0.3, 800., 0.78])
o = softmax(a)
print(o)
