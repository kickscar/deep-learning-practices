# coding: utf-8
# 최소제곱법(Method of Least Squares)
import os
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import method_least_squares
except ImportError:
    print('Library Module Can Not Found')


# data
times = np.array([2, 4, 6, 8])
scores = np.array([81, 93, 91, 97])

# a, b 구하기
a, b = method_least_squares(times, scores)

# 결과
print(f'직선 y = {a}x + {b}')
scores_predict = a * times + b

# 그래프
fig, subplots = plt.subplots()
subplots.scatter(times, scores)
subplots.plot(times, scores_predict, 'ro-')
plt.show()
