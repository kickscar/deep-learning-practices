# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, numerical_gradient
except ImportError:
    raise ImportError("Library Module Can Not Found")


# wrapper for function of cross entropy error
def loss(w, x, t):
    z = np.dot(x, w)
    y = softmax(z)
    e = cross_entropy_error(y, t)
    return e


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

w = np.random.randn(2, 3)

gradient = numerical_gradient(loss, w, (x, t))
print(gradient)
