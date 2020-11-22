# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient): Parameter 가중치(w) 편미분 과정 #1
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, numerical_gradient2
except ImportError:
    raise ImportError("Library Module Can Not Found")


x = np.array([0.6, 0.9])    # 입력 (input)
t = np.array([0, 0, 1])     # 정답 label(target)


def loss(w):
    """

    :param w: 가중치 매개변수(parameter weight)
    :return: 오차(error)
    """
    z = np.dot(x, w)
    y = softmax(z)
    e = cross_entropy_error(y, t)
    
    return e


_w = np.random.randn(2, 3)
print(_w)

gradient = numerical_gradient2(loss, _w)
print(gradient)
