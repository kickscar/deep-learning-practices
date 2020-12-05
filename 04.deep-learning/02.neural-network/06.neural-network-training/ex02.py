# coding: utf-8
# 신경망 학습: 교차 엔트로피 손실함수 (Cross Entropy Error, CEE)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")

# data
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.])
y2 = np.array([0.1, 0.05, 0.05, 0.6, 0.02, 0.03, 0.05, 0.1, 0., 0.])
y3 = np.array([0., 0., 0.95, 0.02, 0.01, 0.01, 0., 0.01, 0., 0.])

# print(np.sum(y1))
# print(np.sum(y2))
# print(np.sum(y3))

# test1
print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))
print(cross_entropy_error(y3, t))


# test2: if batch_size is 3
train_t_batch = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

y = np.array([
    [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.],
    [0.05, 0.05, 0.03, 0.07, 0.5, 0.03, 0.02, 0.1, 0.08, 0.07],
    [0.1, 0.03, 0.05, 0., 0.02, 0.7, 0., 0.1, 0., 0.]
])

print(cross_entropy_error(y, train_t_batch))



