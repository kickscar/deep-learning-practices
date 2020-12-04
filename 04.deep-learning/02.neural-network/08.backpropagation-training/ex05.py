# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Test: SGD based on Backpropagation Gradient
import pickle
import shutil
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
network.initialize(input_size=train_x.shape[1], hidden_size=[50, 100], output_size=train_t.shape[1])

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
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
    gradient = network.backpropagation_gradient(train_x_batch, train_t_batch)
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

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append(test_accuracy)

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000:.4f}ms - loss:{loss:.4f} - accuracy: (train, test)=({train_accuracy:.3f}, {test_accuracy:.3f})')

        elapsed = 0


# 5. save model
print(f'\nsaving model.....', end='')

model_directory = os.path.join(os.getcwd(), 'model')
if os.path.exists(model_directory):
    shutil.rmtree(model_directory)

os.mkdir(model_directory)

model_file = os.path.join(model_directory, 'model.pkl')
train_losses_file = os.path.join(model_directory, 'train_loss.pkl')
train_accuracy_file = os.path.join(model_directory, 'train_accuracy.pkl')
test_accuracy_file = os.path.join(model_directory, 'test_accuracy.pkl')
with open(model_file, 'wb') as f_params,\
        open(train_losses_file, 'wb') as f_train_losses,\
        open(train_accuracy_file, 'wb') as f_train_accuracy,\
        open(test_accuracy_file, 'wb') as f_test_accuracy:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_train_losses, -1)
    pickle.dump(train_accuracies, f_train_accuracy, -1)
    pickle.dump(test_accuracies, f_test_accuracy, -1)

print('done')
