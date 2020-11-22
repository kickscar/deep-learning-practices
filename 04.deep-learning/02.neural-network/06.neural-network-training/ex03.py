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
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
print(train_x.shape)  # 60000 X 784
print(train_t.shape)  # 60000 X 10

# test2
train_size = len(train_x)
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)

train_x_batch = train_x[batch_mask]
train_t_batch = train_t[batch_mask]
print(train_x_batch.shape)
print(train_t_batch.shape)

# test3
# if batch_size is 3
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

# print(np.sum(train_t_batch, axis=1))
# print(np.sum(y, axis=1))

print(cross_entropy_error(y, train_t_batch))



