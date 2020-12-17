# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MNISTNet
# Model Fitting
import pickle
import shutil
import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. graph set-up
fig = plt.figure(figsize=(15, 4))
spec = plt.GridSpec(15, 15, bottom=0.08, top=0.95, left=0.05, right=0.98, hspace=0, wspace=1.6)

ax1 = fig.add_subplot(spec[:-1, :7], xlabel='Iterations', ylabel='Loss')
lineLosses = ax1.plot([])[0]
ax1.grid()
ax1.set_xlim(0, train_x.shape[0])
ax1.set_ylim(0., 3.)

ax2 = fig.add_subplot(spec[:-1, 7:-1], xlabel='Epochs', ylabel='Loss - Accuracy')

line1 = ax2.plot([])[0]
line1.set_marker('.')
line1.set_color('red')
line1.set_label('Train Accuracy')

line2 = ax2.plot([])[0]
line2.set_marker('.')
line2.set_color('red')
line2.set_label('Train Loss')

line3 = ax2.plot([])[0]
line3.set_marker('.')
line3.set_color('blue')
line3.set_label('Test Accuracy')

line4 = ax2.plot([])[0]
line4.set_marker('.')
line4.set_color('blue')
line4.set_label('Test Loss')

ax2.legend(handles=[line1, line2, line3, line4], loc='best')
ax2.set_xticks(range(1, 21))
ax2.set_ylim(0., 1.)

ax3 = fig.add_subplot(spec[:3, -1], xlabel='Train Image')
# ax3.axis("off")
ax3.set_xticks([])
ax3.set_yticks([])

plt.draw()
plt.pause(0.01)

# 2. hyperparamters
epochs, batch_size = 20, 100
learning_rate = 0.1

# 4. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50, 100], output_size=output_size)

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

elapsed, epoch_idx = 0, 0
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
losses = []

for idx in range(1, iterations+1):
    # 4-1. stopwatch: start
    stime = time.time()

    # 4-2. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    # image view
    ax3.imshow((train_x_batch[99].reshape(28, 28) * 255).astype(np.uint8), cmap='gray')

    # 4-3. gradient
    gradient = network.backpropagation_gradient(train_x_batch, train_t_batch)

    # 4-4. update parameters
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    #
    loss = network.loss(test_x, test_t)
    losses.append((idx+1, loss))
    lineLosses.set_data(zip(*losses))

    # 4-5. stopwatch: stop
    elapsed += (time.time() - stime)

    # 4-5. epoch history
    if idx % epoch_size == 0:
        loss = network.loss(train_x, train_t)
        history['loss'].append((epoch_idx, loss))
        line1.set_data(zip(*history['loss']))

        accuracy = network.accuracy(train_x, train_t)
        history['accuracy'].append((epoch_idx, accuracy))
        line2.set_data(zip(*history['accuracy']))

        val_loss = network.loss(test_x, test_t)
        history['val_loss'].append((epoch_idx, val_loss))
        line3.set_data(zip(*history['val_loss']))

        val_accuracy = network.accuracy(test_x, test_t)
        history['val_accuracy'].append((epoch_idx, val_accuracy))
        line4.set_data(zip(*history['val_accuracy']))

        epoch_idx += 1

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed:.3f}s - loss:{loss:.4f} - accuracy:{accuracy:.4f}')

        elapsed = 0

    plt.draw()
    plt.pause(0.00000000000001)

plt.show()

# 5. save model
# print(f'\nsaving model & history.....', end='')
#
# model_directory = os.path.join(Path(os.getcwd()).parent, 'model')
# model_file = os.path.join(model_directory, 'model.pkl')
# history_file = os.path.join(model_directory, 'history.pkl')
#
# if os.path.exists(model_directory):
#     shutil.rmtree(model_directory)
#
# os.mkdir(model_directory)
#
# with open(model_file, 'wb') as f_model, open(history_file, 'wb') as f_history:
#     pickle.dump(network.params, f_model, -1)
#     pickle.dump(history, f_history, -1)
#
# print('done')


