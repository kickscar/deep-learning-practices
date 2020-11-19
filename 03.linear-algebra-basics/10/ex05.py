# coding: utf-8
# 다중선형회귀(수치미분)
import os
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import mean_squares_error, gradient_descent
except ImportError:
    print('Library Module Can Not Found')


# data
times = np.array([2, 4, 6, 8])
ptimes = np.array([0, 4, 2, 3])
scores = np.array([81, 93, 91, 97])


# 경사하강법
params = gradient_descent(
    mean_squares_error,
    np.array([0., 0., 0.]),
    epoch=3000,
    data_in=np.array([times, ptimes]),
    data_out=scores)


# 3D 그래프
axes = plt.axes(projection='3d')
axes.scatter(times, ptimes, scores)
axes.set_xlabel('Study Hours')
axes.set_ylabel('Private Class Hours')
axes.set_zlabel('Score')

plt.show()

# predict!!
x1_p = 2
x2_p = 2
y_p = params[0] * x1_p + params[1] * x2_p + params[2]
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을 때, 받을 수 있는 점수: {int(y_p)}점')
