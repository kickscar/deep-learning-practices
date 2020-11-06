# coding: utf-8
# 최소제곱법(Method of Least Squares)
import numpy as np
from matplotlib import pyplot as plt


def mls(x, y):
    mx = np.mean(x)
    my = np.mean(y)

    mls_a = sum([(i - mx) * (j - my) for i, j in tuple(zip(x, y))]) / sum([(i - mx) ** 2 for i in x])
    mls_b = my - (mx * mls_a)

    return mls_a, mls_b


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 기울기 a, y절편 b 구하기
a, b = mls(times, scores)

# 결과
print(f'직선 y = {a}x + {b}')
scores_predict = [(a * i) + b for i in times]

fig, subplots = plt.subplots()
subplots.scatter(times, scores)
subplots.plot(times, scores_predict, 'ro-')
plt.show()
