# coding: utf-8
# 신경망 기울기(Neural Network Gradient): Parameter 가중치(w) 편미분 과정 #1
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


def loss(w, x, t):
    """
    :param w: 가중치 매개변수(parameter weight)
    :param x: 입력 (input)
    :param t: 정답 label(target)
    :return: 오차(error)
    """
    z = np.dot(x, w)
    y = common.softmax(z)
    e = common.cross_entropy_error(y, t)

    return e


# Numerical Gradient
# loss(손실함수), param(w), x, t를 파라미터로 받는 기울기 함수
def numerical_gradient(f, param, x, t):
    h = 1e-4
    gradient = np.zeros_like(param)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        x[idx] = float(tmp) + h
        h1 = f(param, x, t)

        x[idx] = float(tmp) - h
        h2 = f(param, x, t)

        gradient[idx] = (h1 - h2) / (2 * h)

        x[idx] = tmp
        it.iternext()

    return gradient


_x = np.array([0.6, 0.9])   # input (x)         2 vector
_t = np.array([0, 0, 1])    # label (one-hot)   3 vector
_w = np.random.randn(2, 3)  # weight            2 x 3 matrix

g = numerical_gradient(loss, _w, _x, _t)
print(g)
