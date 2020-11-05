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


def gradient_descent(f, x, lr=0.01, epoch=100):
    for i in range(epoch):
        gradient = numerical_gradient(f, x)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x


gradient_descent(function, np.array([-3., 4.]), 0.1)
# gradient_descent(function, np.array([-3., 4.]), 10)
# gradient_descent(function, np.array([-3., 4.]), 1e-10)

