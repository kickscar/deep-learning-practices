# coding: utf-8
# multi-layer 01.perceptron
import os
import sys

try:
    sys.path.append(os.getcwd())
    from ex02 import AND
    from ex03 import NAND
    from ex04 import OR
except ImportError:
    raise ImportError("lib.mnist Module Can't Not Found")


def identity_func(x):
    return x


def XOR(x1, x2):
    a1 = NAND(x1, x2)
    a2 = OR(x1, x2)

    return identity_func(AND(a1, a2))


print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

