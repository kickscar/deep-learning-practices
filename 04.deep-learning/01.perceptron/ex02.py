# coding: utf-8
# and gate: Perceptron
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(x*w) + b

    return 1 if y > 0 else 0


if __name__ == '__main__':
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))
