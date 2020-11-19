# coding: utf-8
import os
import sys
import numpy as np
try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'lib'))

    from ex02 import AND
    from ex03 import NAND
    from ex04 import OR
    from common import identity
except ImportError:
    raise ImportError("Modules Can Not Found")


def XOR(x):
    a1 = NAND(x)
    a2 = OR(x)
    a3 = AND(np.array([a1, a2]))

    # y = a3
    y = identity(a3)

    return y


y = XOR(np.array([0, 0]))
print(y)

y = XOR(np.array([1, 0]))
print(y)

y = XOR(np.array([0, 1]))
print(y)

y = XOR(np.array([1, 1]))
print(y)
