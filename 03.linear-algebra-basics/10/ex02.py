# coding: utf-8
# 평균제곱오차(MSE, Mean Squares Error)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import mean_squares_error, method_least_squares
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 최소제곱법으로 기울기 a, y절편 b 구하기
a, b = method_least_squares(times, scores)

print(f'오차(평균제곱오차):{mean_squares_error(np.array([a, b]), data_training=(times, scores))}')

