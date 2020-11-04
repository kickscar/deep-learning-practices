# 편미분(Numerical Diffirentiation)


def f1(x1):
    return 3.0 ** 2 + x1 ** 2


def analytic_diff_f1(x1):
    return 2 * x1


def numerical_diff(f, x1):
    h = 1e-4
    return (f(x1+h) - f(x1-h)) / (2*h)


print(f'Numerical Diffirentiation Value: {numerical_diff(f1, 4.)}')
print(f'Analytic  Diffirentiation Value: {analytic_diff_f1(4.)}')
