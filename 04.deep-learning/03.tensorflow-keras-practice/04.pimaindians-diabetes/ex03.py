# coding: utf-8
# Training Neural Network
# Data Set: Pima Idians Diabets Data Set
# Network: TwoLayerNet2
# Test: SGD based on Backpropagation Gradient
import pickle
import sys
import os
import time
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    import twolayernet2 as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
dataset = np.loadtxt("./dataset/pimaindians-diabetes.csv", delimiter=",")
train_x = np.array(dataset[:, 0:8])
train_t = np.array(dataset[:, 8])[:, np.newaxis]
train_t = np.c_[train_t, train_t == 0]

# 2. hyperparamters
batch_size = 10
epochs = 200
learning_rate = 0.001

# 3. model frame
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1])

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = round(train_size / batch_size)
iterations = epochs * epoch_size

elapsed = 0
epoch_idx = 0
train_losses = []
train_accuracies = []
test_accuracies = []

for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # 4-2. gradient
    stime = time.time()
    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
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

        train_accuracy = network.accuracy(train_x, train_t)
        train_accuracies.append(train_accuracy)

        print(f'\nEpoch {epoch_idx}/{epochs}')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000}ms - loss:{loss} - [accuracy: (train,) = ({train_accuracy},)]')

        elapsed = 0


# 5. serialize params & train losses
print(f'\ncreating pickle...')

params_file = os.path.join(os.getcwd(), 'model', f'twolayer_params.pkl')
train_losses_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_losses.pkl')
train_accuracy_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_accuracy.pkl')

with open(params_file, 'wb') as f_params,\
        open(train_losses_file, 'wb') as f_train_losses,\
        open(train_accuracy_file, 'wb') as f_train_accuracy:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_train_losses, -1)
    pickle.dump(train_accuracies, f_train_accuracy, -1)
print(f'done!')

