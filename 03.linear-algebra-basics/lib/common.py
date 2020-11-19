import numpy as np
from inspect import signature


# Numerical (Partial) Differentiation
def numerical_diff(f, x, data_in=None, data_out=None):
    h = 1e-4

    # scalar
    if x.ndim == 0:
        return (f(x+h) - f(x-h)) / (2 * h)

    # vector
    gradient = np.zeros_like(x)
    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x) if len(signature(f).parameters) == 1 else f(x, data_in, data_out)

        x[i] = tmp - h
        h2 = f(x) if len(signature(f).parameters) == 1 else f(x, data_in, data_out)

        gradient[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

    return gradient


# Gradient
numerical_gradient = numerical_diff


# Gradient Descent
def gradient_descent(f, x, lr=0.01, epoch=100, data_in=None, data_out=None):
    for i in range(epoch):
        gradient = numerical_gradient(f, x, data_in, data_out)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x


# Mean Squares Error(MSE, 평균제곱오차) -> 과제: lambda 변경
def mean_squares_error(x, data_in, data_out):
    return np.mean((x[:-1] @ (data_in[np.newaxis, :] if data_in.ndim == 1 else data_in) + x[-1:] - data_out)**2)


# Method of Least Squares(MLS, 최소제곱법): 여러 점(독립변수 X, 종속변수 Y)에서 직선의 기울기 구하기
def method_least_squares(x, y):
    mx = np.mean(x)
    my = np.mean(y)

    a = np.sum((x-mx)*(y-my)) / np.sum((x-mx)**2)
    b = my - (mx * a)

    return a, b
