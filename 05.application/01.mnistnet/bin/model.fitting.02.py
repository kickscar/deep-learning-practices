# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MNISTNet
# Model Fitting with Realtime-Graph
import os
import sys
import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, animation
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparamters
epochs, batch_size = 20, 100
learning_rate = 0.1

# 3. model fitting parameters
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

# 4. graph set-up
fig = plt.figure(figsize=(13, 3))
spec = plt.GridSpec(20, 20, bottom=0.13, top=0.95, left=0.05, right=0.98, hspace=0, wspace=1.8)

ax1 = fig.add_subplot(spec[:-1, :10], xlabel='Iterations', ylabel='Loss')
ax2 = fig.add_subplot(spec[:-1, 11:], xlabel='Epochs', ylabel='Loss - Accuracy')

line_iters_loss = ax1.plot([])[0]

line_train_acc = ax2.plot([])[0]
line_train_acc.set_marker('.')
line_train_acc.set_color('red')
line_train_acc.set_label('Train Accuracy')

line_train_loss = ax2.plot([])[0]
line_train_loss.set_marker('.')
line_train_loss.set_color('red')
line_train_loss.set_label('Train Loss')

line_test_acc = ax2.plot([])[0]
line_test_acc.set_marker('.')
line_test_acc.set_color('blue')
line_test_acc.set_label('Test Accuracy')

line_test_loss = ax2.plot([])[0]
line_test_loss.set_marker('.')
line_test_loss.set_color('blue')
line_test_loss.set_label('Test Loss')

ax1.grid()
ax1.set_xlim(0, iterations)
ax1.set_ylim(0., 3.)

ax2.legend(handles=[line_train_acc, line_train_loss, line_test_acc, line_test_loss], loc='best')
ax2.grid()
ax2.set_xlim(0, epochs)
ax2.set_xticks(range(0, epochs+1))
ax2.set_ylim(0., 1.)
ax2.set_yticks(np.arange(0., 1.1, 0.1))

history = {'iters_loss': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}


def update_graph(data):
    if data[0] is not None:
        history['iters_loss'].append(data[0])

    if data[1] is not None:
        history['loss'].append(data[1])

    if data[2] is not None:
        history['accuracy'].append(data[2])

    if data[3] is not None:
        history['val_loss'].append(data[3])

    if data[4] is not None:
        history['val_accuracy'].append(data[4])

    line_iters_loss.set_data(range(0, len(history['iters_loss'])), history['iters_loss'])
    line_train_loss.set_data(range(0, len(history['loss'])), history['loss'])
    line_train_acc.set_data(range(0, len(history['accuracy'])), history['accuracy'])
    line_test_loss.set_data(range(0, len(history['val_loss'])), history['val_loss'])
    line_test_acc.set_data(range(0, len(history['val_accuracy'])), history['val_accuracy'])

    return line_iters_loss, line_train_loss, line_train_acc, line_test_loss, line_test_acc


# 5. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50, 100], output_size=output_size)

# 6. model fitting
def model_fit():
    elapsed, epoch_idx = 0, 0

    for idx in range(1, iterations+1):
        # 4-0.
        history_data = [None, None, None, None, None]

        # 4-1. stopwatch: start
        stime = time.time()

        # 4-2. fetch mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        train_x_batch = train_x[batch_mask]
        train_t_batch = train_t[batch_mask]

        # 4-3. gradient
        gradient = network.backpropagation_gradient(train_x_batch, train_t_batch)

        # 4-4. update parameters
        for key in network.params:
            network.params[key] -= learning_rate * gradient[key]

        # 4-5. stopwatch: stop
        elapsed += (time.time() - stime)

        # 4-5. epoch history
        if idx % epoch_size == 0:
            epoch_idx += 1

            history_data[1] = network.loss(train_x, train_t)
            history_data[2] = network.accuracy(train_x, train_t)
            history_data[3] = network.loss(test_x, test_t)
            history_data[4] = network.accuracy(test_x, test_t)

            print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
            print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed:.3f}s - loss:{history_data[3]:.4f} - accuracy:{history_data[4]:.4f}')

            elapsed = 0

        history_data[0] = history_data[3] or network.loss(test_x, test_t)

        yield history_data


# main
ani = animation.FuncAnimation(fig, update_graph, model_fit, interval=1, blit=True, save_count=0)
plt.show()
