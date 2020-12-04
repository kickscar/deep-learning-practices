# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Test: SGD based on Numerical Gradient
import sys
import os
import time
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
batch_size = 100
epochs = 30
learning_rate = 0.1

# 3. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50], output_size=output_size)

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = 1  # epochs * epoch_size

elapsed, epoch_idx = 0, 0
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

for idx in range(1, iterations+1):
    # 4-1. stopwatch: start
    stime = time.time()

    # 4-2. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-3. gradient
    gradient = network.numerical_gradient(train_x_batch, train_t_batch)

    # 4-4. update parameters
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-5. stopwatch: stop
    elapsed += (time.time() - stime)

    # 4-6. print a loss
    loss = network.loss(train_x, train_t)
    print(f'#{idx}: loss:{loss:.4f} : elapsed time[{elapsed:.3f}s]')
