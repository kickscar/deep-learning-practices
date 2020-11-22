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


def loss(w, x, t):
    """

    :param w: 가중치 매개변수(parameter weight)
    :param x: 입력 (input)
    :param t: 정답 label(target)
    :return: 오차(error)
    """
    z = np.dot(x, w)
    y = softmax(z)
    e = cross_entropy_error(y, t)

    return e


_x = np.array([0.6, 0.9])
_t = np.array([0, 0, 1])
_w = np.random.randn(2, 3)
print(_w)

gradient = numerical_gradient(loss, _w, _x, _t)
print(gradient)
