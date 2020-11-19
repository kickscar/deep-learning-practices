# coding: utf-8
# 경사하강법(수치미분)
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import mean_squares_error, gradient_descent
except ImportError:
    print('Library Module Can Not Found')

# data
times = np.array([2, 4, 6, 8])
scores = np.array([81, 93, 91, 97])

# 경사하강법
params = gradient_descent(mean_squares_error, np.array([0., 0.]), epoch=5000, data_in=times, data_out=scores)
print(params)

# 평균제곱오차
e = mean_squares_error(params, data_in=times, data_out=scores)
print(f'오차:{e}')
