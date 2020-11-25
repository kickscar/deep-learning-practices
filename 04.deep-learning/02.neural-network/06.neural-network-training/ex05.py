# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
import pickle
import sys
import os
import time
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
numiters = 1  # 10000
szbatch = 100
sztrain = train_x.shape[0]
ratelearning = 0.1

# 3. initialize network
network.initialize(sz_input=train_x.shape[1], sz_hidden=50, sz_output=train_t.shape[1])

# 4. training
train_losses = []

for idx in range(numiters):
    #
    start = time.time()

    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)

    # 4-3. update parameters
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    #
    end = time.time()

    print(f'#{idx+1}: loss:{loss} : elapsed time[{end - start} secs]')


# 5. save params (serialize)
save_params_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_params.pkl')
network.save_params(save_params_file)

# 6. save train loss (serialize)
save_train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_train_loss.pkl')
print(f'Creating Pickle({save_train_loss_file}) file ...')
with open(save_train_loss_file, 'wb') as f:
    pickle.dump(train_losses, f, -1)
print("Done!")

