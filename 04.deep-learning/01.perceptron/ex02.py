# coding: utf-8
# and gate: Perceptron
import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int);


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    return step_function(np.sum(x*w) + b)


if __file__ == 'main':
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))
