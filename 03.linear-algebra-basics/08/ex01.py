# 기울기(Gradient)
import numpy as np


def function(x):
    return np.sum(x**2, axis=0)


def numerical_gradient(f, x):
    h = 1e-4

    gradient = np.zeros_like(x)
    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x)

        x[i] = tmp - h
        h2 = f(x)

        gradient[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

    return gradient


print(numerical_gradient(function, np.array([3., 4.])))
print(numerical_gradient(function, np.array([-1., -1.5])))
print(numerical_gradient(function, np.array([-0.25, -0.25])))
