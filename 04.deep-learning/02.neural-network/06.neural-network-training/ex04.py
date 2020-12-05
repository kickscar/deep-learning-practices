# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
# Model Fitting(Numerical Gradient + SGD)
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
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")


# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
epochs, batch_size = 20, 100
learning_rate = 0.1

# 3. initialize network
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1])

# 4. training
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

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
    elapsed = time.time() - stime

    # 4-6. epoch accuracy
    if idx % epoch_size == 0:
        epoch_idx += 1

        loss = network.loss(train_x, train_t)
        history['loss'].append(loss)

        accuracy = network.accuracy(train_x, train_t)
        history['accuracy'].append(accuracy)

        val_loss = network.loss(test_x, test_t)
        history['val_loss'].append(val_loss)

        val_accuracy = network.accuracy(test_x, test_t)
        history['val_accuracy'].append(val_accuracy)

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed:.3f}s - loss:{loss:.4f} - accuracy:{accuracy:.4f}')

        elapsed = 0

    # 4-7. print out batch loss
    loss_batch = network.loss(train_x_batch, train_t_batch)
    print(f'#{idx}: batch loss:{loss_batch} : elapsed time[{elapsed}s]')


# 5. save model
print(f'\nsaving model & history.....', end='')

model_directory = os.path.join(os.getcwd(), 'model')
model_file = os.path.join(model_directory, 'model.pkl')
history_file = os.path.join(model_directory, 'history.pkl')

if os.path.exists(model_directory):
    shutil.rmtree(model_directory)

os.mkdir(model_directory)

with open(model_file, 'wb') as f_model, open(history_file, 'wb') as f_history:
    pickle.dump(network.params, f_model, -1)
    pickle.dump(history, f_history, -1)

print('done')

