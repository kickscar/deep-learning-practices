# coding: utf-8
# 3층 신경망 구현하기 – 모두 합치기
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, identity
except ImportError:
    raise ImportError("Library Module Can Not Found")


def init_network():
    return {
        'w1': np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 1.]]),
        'b1': np.array([0.1, 0.2, 0.3]),
        'w2': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b2': np.array([0.1, 0.2]),
        'w3': np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3': np.array([0.1, 0.2])
    }


def forward_propagation(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(w1, x) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(w2, z1) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(w3, z2) + b3
    z3 = identity(a3)

    return z3


network = init_network()
x = np.array([1., 5.])
y = forward_propagation(network, x)

print(y)
