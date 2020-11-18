# coding: utf-8
# 최소제곱법(Method of Least Squares)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import method_least_squares
except ImportError:
    print('Library Module Can Not Found')
from matplotlib import pyplot as plt


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 기울기 a, y절편 b 구하기
a, b = method_least_squares(times, scores)

# 결과
print(f'직선 y = {a}x + {b}')
scores_predict = [(a * i) + b for i in times]

fig, subplots = plt.subplots()
subplots.scatter(times, scores)
subplots.plot(times, scores_predict, 'ro-')
plt.show()
