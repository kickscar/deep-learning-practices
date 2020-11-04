# 수치미분(Numerical Diffirentiation) VS 해석미분(Analytic Diffirentiation)


def f1(x):
    return 20*(x-2)**2 + 500


def analytic_diff_f1(x):
    return 40*x - 80


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


print(f'Numerical Diffirentiation Value: {numerical_diff(f1, 5)}')
print(f'Analytic  Diffirentiation Value: {analytic_diff_f1(5)}')
