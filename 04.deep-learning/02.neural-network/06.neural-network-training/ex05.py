# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
import datetime
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
numiters = 12000
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
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
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

    print(f'#{idx}: loss:{loss} : elapsed time[{elapsed}s]')


# 5. serialize params & train losses
print(f'creating pickle...')
now = datetime.datetime.now()

params_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_params_{now:%Y%m%d%H%M%S}.pkl')
train_losses_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_losses_{now:%Y%m%d%H%M%S}.pkl')
train_accuracy_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_accuracy_{now:%Y%m%d%H%M%S}.pkl')
test_accuracy_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_test_accuracy_{now:%Y%m%d%H%M%S}.pkl')

with open(params_file, 'wb') as f_params,\
        open(train_losses_file, 'wb') as f_train_losses,\
        open(train_accuracy_file, 'wb') as f_train_accuracy,\
        open(test_accuracy_file, 'wb') as f_test_accuracy:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_train_losses, -1)
    pickle.dump(train_accuracies, f_train_accuracy, -1)
    pickle.dump(test_accuracies, f_test_accuracy, -1)
print(f'done!')

