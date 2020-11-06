# coding: utf-8
# 평균제곱오차(MSE, Mean Squares Error)
import numpy as np


def mls(x, y):
    mx = np.mean(x)
    my = np.mean(y)

    mls_a = sum([(i - mx) * (j - my) for i, j in tuple(zip(x, y))]) / sum([(i - mx) ** 2 for i in x])
    mls_b = my - (mx * mls_a)

    return mls_a, mls_b


def mse(x, data_x, data_y):
    data_y_hat = [x[0] * dx + x[1] for dx in data_x]
    e = np.mean([(dyh - dy) ** 2 for dyh, dy in tuple(zip(data_y_hat, data_y))])
    return e


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 최소제곱법으로 기울기 a, y절편 b 구하기
a, b = mls(times, scores)

print(f'오차(평균제곱오차):{mse(np.array([a, b]), times, scores)}')

