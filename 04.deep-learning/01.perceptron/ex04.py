# coding: utf-8
# nand gate: Perceptron
import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int);


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    return step_function(np.sum(x*w) + b)


if __file__ == 'main':
    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1))
