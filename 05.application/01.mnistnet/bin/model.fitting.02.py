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
    raise ImportError("modules can not be found")


def graph_setup(ax1_xlims=0, ax2_xlims=0):
    fig = plt.figure(figsize=(13, 3))
    spec = plt.GridSpec(20, 20, bottom=0.13, top=0.95, left=0.05, right=0.98, hspace=0, wspace=1.8)

    ax1 = fig.add_subplot(spec[:-1, :10], xlabel='Iterations', ylabel='Loss')
    ax2 = fig.add_subplot(spec[:-1, 11:], xlabel='Epochs', ylabel='Loss - Accuracy')

    graphs = [
        {'line2d': ax1.plot([])[0], 'data': []},
        {'line2d': ax2.plot([])[0], 'data': []},
        {'line2d': ax2.plot([])[0], 'data': []},
        {'line2d': ax2.plot([])[0], 'data': []},
        {'line2d': ax2.plot([])[0], 'data': []}
    ]

    graphs[1]['line2d'].set_marker('.')
    graphs[1]['line2d'].set_color('red')
    graphs[1]['line2d'].set_label('Train Accuracy')

    graphs[2]['line2d'].set_marker('.')
    graphs[2]['line2d'].set_color('red')
    graphs[2]['line2d'].set_label('Train Loss')

    graphs[3]['line2d'].set_marker('.')
    graphs[3]['line2d'].set_color('blue')
    graphs[3]['line2d'].set_label('Test Accuracy')

    graphs[4]['line2d'].set_marker('.')
    graphs[4]['line2d'].set_color('blue')
    graphs[4]['line2d'].set_label('Test Loss')

    ax1.grid()
    ax1.set_xlim(0, ax1_xlims)
    ax1.set_ylim(0., 3.)

    ax2.legend(handles=[graphs[1]['line2d'], graphs[2]['line2d'], graphs[3]['line2d'], graphs[4]['line2d']], loc='best')
    ax2.grid()
    ax2.set_xlim(0, ax2_xlims)
    ax2.set_xticks(range(0, ax2_xlims+1))
    ax2.set_ylim(0., 1.)
    ax2.set_yticks(np.arange(0., 1.1, 0.1))

    return fig, graphs


def graph_update(data, graphs):

    lines = [None] * len(data)

    for i, d in enumerate(data):

        graph_data = graphs[i]['data']
        graph_line = graphs[i]['line2d']

        if d is not None:
            graph_data.append(d)

        graph_line.set_data(range(len(graph_data)), graph_data)
        lines[i] = graph_line

    return lines


def print_progress(cur_progress, max_progress=100, length=100, prefix='', suffix='', fill='█'):
    # percent = int(100 * (cur_progress / float(max_progress)))

    filled_length = int(length * cur_progress // max_progress)
    bar = fill * filled_length + '-' * (length - filled_length)

    # print(f'\r{prefix} |{bar}| {percent:3d}% {suffix}', end='\r')
    print(f'{prefix} |{bar}| {suffix}', end='\r')

    cur_progress == max_progress and print('')


def model_fit():
    # dataset
    (train_x, train_t), (test_x, test_t) = model_fit.dataset

    # hyperparameters
    epochs, batch_size, learning_rate = model_fit.hyperparams

    # model fitting parameters
    train_size, input_size = train_x.shape
    epoch_size = int(train_size / batch_size)
    iterations = epochs * epoch_size

    # stopwatch: start
    elapsed, epoch_idx = 0, 0
    stime = time.time()

    for idx in range(1, iterations+1):
        # init history
        history = [None] * 5

        # fetch mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        train_x_batch = train_x[batch_mask]
        train_t_batch = train_t[batch_mask]

        # gradient
        gradient = network.backpropagation_gradient(train_x_batch, train_t_batch)

        # update parameters
        for key in network.params:
            network.params[key] -= learning_rate * gradient[key]

        # iteration history
        history[0] = network.loss(test_x, test_t)

        # epoch history
        suffix = ''
        if idx % epoch_size == 0:
            history[1] = network.loss(train_x, train_t)
            history[2] = network.accuracy(train_x, train_t)
            history[3] = history[0]
            history[4] = network.accuracy(test_x, test_t)

            # stopwatch: stop
            elapsed += (time.time() - stime)

            suffix = f' - {elapsed:.3f}s - loss:{history[3]:.4f} - accuracy:{history[4]:.4f}'

            # stopwatch: start
            elapsed = 0
            stime = time.time()

        elif idx % epoch_size == 1:
            epoch_idx += 1
            print(f'Epoch {epoch_idx:2d}/{epochs:2d}')

        idx_in_epoch = int(idx / epoch_idx)
        prefix = f'{idx_in_epoch:3d}/{epoch_size:3d}'
        print_progress(idx_in_epoch, max_progress=epoch_size, length=60, prefix=prefix, suffix=suffix)

        yield history


if __name__ == '__main__':

    # 1. load training/test data
    model_fit.dataset = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    # 2. hyperparamters for model fitting
    model_fit.hyperparams = (20, 100, 0.1)  # (epochs, batch_size, learning_rate)

    # 3. graph set-up
    iters = model_fit.hyperparams[0] * int(model_fit.dataset[0][0].shape[0] / model_fit.hyperparams[1])
    f, g = graph_setup(ax1_xlims=iters, ax2_xlims=model_fit.hyperparams[0])

    # 4. model frame
    network.initialize(input_size=model_fit.dataset[0][0].shape[1], hidden_sizes=(50, 100), output_size=model_fit.dataset[0][1].shape[1])

    # 5. realtime graph with model fitting
    ani = animation.FuncAnimation(
        f,
        graph_update,
        model_fit,
        fargs=(g, ),
        init_func=lambda: None,
        interval=1,
        blit=False,
        repeat=False,
        save_count=0)
    plt.show()
