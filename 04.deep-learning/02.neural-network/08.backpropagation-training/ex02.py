# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet2
# Test: SGD based on Backpropagation Gradient
import sys
import os
import time
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet2 as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
batch_size = 100
epochs = 30
learning_rate = 0.1

train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = 1  # epochs * epoch_size

# 3. model frame
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1])

# 4. model fitting
for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    stime = time.time()
    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
    elapsed = time.time() - stime

    # 4-3. update parameters
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    print(f'#{idx}: loss:{loss} : elapsed time[{elapsed*1000}ms]')


