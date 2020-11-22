# coding: utf-8
# 신경망 학습: 신경망 기울기(Neural Network Gradient): Parameter 가중치(w) 편미분 과정 #4
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, numerical_gradient2
except ImportError:
    raise ImportError("Library Module Can Not Found")

x = np.array([0.6, 0.9])                    # input (x)         2 vector
t = np.array([0, 0, 1])                     # label (one-hot)   3 vector
params = {
    'w1': np.random.randn(2, 3),            # weight            2 x 3 matrix
    'b1': np.array([0.45, 0.23, 0.11])      # bias              3 vector
}


def foward_propagation():
    w1 = params['w1']
    b1 = params['b1']

    z = np.dot(x, w1) + b1
    y = softmax(z)

    return y


def loss(w):
    """

    :param w: dummy parameter(아무짝에 필요없는..., 편미분 함수 내부 콜 형식때문에... --;)
    :return: 오차(error)
    """
    y = foward_propagation()
    e = cross_entropy_error(y, t)

    return e


def numerical_gradient_net():
    """
    :return: 각 parameter 에 대한 편미분 결과(기울기)
    """
    return {
        'w1': numerical_gradient2(loss, params['w1']),
        'b1': numerical_gradient2(loss, params['b1'])
    }


g = numerical_gradient_net()
print(g)
