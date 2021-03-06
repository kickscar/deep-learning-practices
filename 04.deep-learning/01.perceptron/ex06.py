# coding: utf-8
# multi-layer perceptron I
import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd())
    from ex02 import AND
    from ex03 import NAND
    from ex04 import OR
except ImportError:
    raise ImportError("Modules Can Not Found")


def XOR(x):
    a1 = NAND(x)
    a2 = OR(x)
    a3 = AND(np.array([a1, a2]))

    y = a3

    return y


y1 = XOR(np.array([0, 0]))
print(y1)

y2 = XOR(np.array([1, 0]))
print(y2)

y3 = XOR(np.array([0, 1]))
print(y3)

y4 = XOR(np.array([1, 1]))
print(y4)
