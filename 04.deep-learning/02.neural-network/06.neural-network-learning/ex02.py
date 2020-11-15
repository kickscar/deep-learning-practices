# coding: utf-8
# 신경망 학습: 미니 배치(mini-batch)
import sys
import os
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    from lib.mnist import load_mnist
except ImportError:
    raise ImportError("lib.mnist Module Can Not Found")


# support only for one-hot
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


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
t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.]
print(cross_entropy_error(np.array(y1), np.array(t1)))


# test4
t2 = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
y2 = [[0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.], [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.]]
print(cross_entropy_error(np.array(y2), np.array(t2)))
