# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: HousingPriceNet
# Test: SGD based on Backpropagation Gradient
import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    import HousingPriceNet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
df = pd.read_csv("./dataset/housing.csv", delim_whitespace=True, header=None)
dataset = df.values
x = dataset[:, 0:13]
t = dataset[:, 13]
train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=3)

train_t = train_t[:, np.newaxis]
test_t = test_t[:, np.newaxis]

# 2. hyperparamters
batch_size = 10
epochs = 50
learning_rate = 0.01

# 3. model frame
network.initialize(input_size=train_x.shape[1], hidden_size=30, output_size=train_t.shape[1])

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

elapsed = 0
epoch_idx = 0
train_losses = []

for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    stime = time.time()
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
    elapsed += (time.time() - stime)

    # 4-3. update parameters
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # 4-5. epoch accuracy
    if idx % epoch_size == 0:
        epoch_idx += 1

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000:.4f}ms - loss:{loss:.4f}')

        elapsed = 0

