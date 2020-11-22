# coding: utf-8
# 신경망 학습: 교차 엔트로피 손실함수 (Cross Entropy Error, CEE)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error_non_batch
except ImportError:
    raise ImportError("Library Module Can Not Found")

# data
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y1 = [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.]
y2 = [0.1, 0.05, 0.05, 0.6, 0.02, 0.03, 0.05, 0.1, 0., 0.]
y3 = [0., 0., 0.95, 0.02, 0.01, 0.01, 0., 0.01, 0., 0.]

# print(np.sum(y1))
# print(np.sum(y2))
# print(np.sum(y3))

# test
print(cross_entropy_error_non_batch(np.array(y1), np.array(t)))
print(cross_entropy_error_non_batch(np.array(y2), np.array(t)))
print(cross_entropy_error_non_batch(np.array(y3), np.array(t)))


