# coding: utf-8
# or gate: Perceptron
import os
import sys
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    from common import step
except ImportError:
    raise ImportError("Modules Can Not Found")


def OR(x):
    w, b = np.array([0.5, 0.5]), -0.2
    a = np.sum(x*w) + b

    # z = 1 if a > 0 else 0
    z = step(a)

    return z


if __name__ == '__main__':
    y = OR(np.array([0, 0]))
    print(y)

    y = OR(np.array([1, 0]))
    print(y)

    y = OR(np.array([0, 1]))
    print(y)

    y = OR(np.array([1, 1]))
    print(y)
