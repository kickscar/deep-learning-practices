# coding: utf-8
# 편미분(Partial Diffirentiation)


def f0(x0):
    return x0 ** 2 + 4.0 ** 2


def analytic_diff_f1(x0):
    return 2 * x0


def numerical_diff(f, x0):
    h = 1e-4
    return (f(x0+h) - f(x0-h)) / (2*h)


print(f'Numerical Diffirentiation Value: {numerical_diff(f0, 3.)}')
print(f'Analytic  Diffirentiation Value: {analytic_diff_f1(3.)}')
