# coding: utf-8
# 수치미분(Numerical Diffirentiation)
# 이차함수


def f1(x):
    return 20*(x-2)**2 + 500


def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h


print(f'Diffirentiation Value: {numerical_diff(f1, 5)}')
