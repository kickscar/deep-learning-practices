# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from two_layer_net import TwoLayerNet
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
numiters = 100  # 10000
szbatch = 100
sztrain = train_x.shape[0]
ratelearning = 0.1

# 3. initialize network
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 4. training
for idx in range(numiters):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    gradient = network.numerical_gradient(train_x_batch, train_t_batch)

    # 4-3. update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= ratelearning * gradient[key]

    # Training Result
    loss = network.loss(train_x_batch, train_t_batch)
    print(f'#{idx+1}: loss:{loss}')
