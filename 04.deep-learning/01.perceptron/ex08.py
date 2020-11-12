# coding: utf-8
import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def identity_func(x):
    return x


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(x*w) + b

    return 1 if y > 0 else 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(x*w) + b

    return 1 if y > 0 else 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    y = np.sum(x*w) + b

    return 1 if y > 0 else 0


def XOR(x1, x2):
    a1 = NAND(x1, x2)
    a2 = OR(x1, x2)

    return AND(a1, a2)


print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
