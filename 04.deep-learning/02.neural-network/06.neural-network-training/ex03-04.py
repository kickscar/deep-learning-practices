# coding: utf-8
# 신경망 기울기(Neural Network Gradient): Parameter 가중치(w) 편미분 과정 #4
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import common
except ImportError:
    raise ImportError("Library Module Can Not Found")

np.random.seed(0)
w = np.random.randn(2, 3)  # weight            2 x 3 matrix


def foward_propagation(x):
    z = np.dot(x, w)
    y = common.softmax(z)

    return y


def loss(dummy):
    y = foward_propagation(loss.x)
    e = common.cross_entropy_error(y, loss.t)

    return e


# Numerical Gradient Wrapper common.numerical_gradient
# x, t를 파라미터로 받는 기울기 함수
def numerical_gradient(x, t):

    loss.x = x
    loss.t = t

    gradient = common.numerical_gradient(loss, w)
    return gradient


_x = np.array([0.6, 0.9])   # input (x)         2 vector
_t = np.array([0, 0, 1])    # label (one-hot)   3 vector

g = numerical_gradient(_x, _t)
print(g)
