# 기울기 구하기

import numpy as np


def function(x):
    return np.sum(x**2, axis=0)


def numerical_diff(f, x):
    h = 1e-4

    gradient = np.zeros_like(x)
    for i in range(x.size()):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x)

        x[i] = tmp - h
        h2 = f(x)


    return np.array([(f(x[i]+h) - f(x[i]-h)) / (2*h) for i in range(x.size())])


print(numerical_diff(function, np.array([3., 4.])))
