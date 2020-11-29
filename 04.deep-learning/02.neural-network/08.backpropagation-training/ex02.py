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
numiters = 1    # 12000
szbatch = 100
sztrain = train_x.shape[0]
szepoch = sztrain / szbatch
ratelearning = 0.1

# 3. initialize network
network.initialize(szinput=train_x.shape[1], szhidden=50, szoutput=train_t.shape[1])

# 4. training
train_losses = []
train_accuracies = []
test_accuracies = []

for idx in range(1, numiters+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    stime = time.time()
    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
    elapsed = time.time() - stime

    # 4-3. update parameters
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # 4-5. epoch accuracy
    if idx % szepoch == 0:
        train_accuracy = network.accuracy(train_x, train_t)
        train_accuracies.append(train_accuracy)

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append(test_accuracy)

    print(f'#{idx}: loss:{loss} : elapsed time[{elapsed} secs]')


