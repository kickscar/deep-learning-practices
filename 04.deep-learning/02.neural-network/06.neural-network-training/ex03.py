# coding: utf-8
# 신경망 학습: 미니 배치(mini-batch)
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


# test1
(x_train, t_train), (x_test, l_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

print(x_train.shape)  # 60000 X 784
print(t_train.shape)  # 60000 X 10

# test2
train_size = len(x_train)
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)

x_train_batch = x_train[batch_mask]
t_train_batch = t_train[batch_mask]
print(x_train_batch.shape)
print(t_train_batch.shape)

# test3
t = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]

y = [
    [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.],
    [0.05, 0.05, 0.03, 0.07, 0.5, 0.03, 0.02, 0.1, 0.08, 0.07],
    [0.1, 0.03, 0.05, 0., 0.02, 0.7, 0., 0.1, 0., 0.]
]

# print(np.sum(t, axis=1))
# print(np.sum(y, axis=1))

print(cross_entropy_error(np.array(y), np.array(t)))



