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

graphs = []


def graph_setup(iterations, epochs):
    fig = plt.figure(figsize=(13, 3))
    spec = plt.GridSpec(20, 20, bottom=0.13, top=0.95, left=0.05, right=0.98, hspace=0, wspace=1.8)

    ax1 = fig.add_subplot(spec[:-1, :10], xlabel='Iterations', ylabel='Loss')
    ax2 = fig.add_subplot(spec[:-1, 11:], xlabel='Epochs', ylabel='Loss - Accuracy')

    graphs.append({'line2d': ax1.plot([])[0], 'data': []})
    graphs.append({'line2d': ax2.plot([])[0], 'data': []})
    graphs.append({'line2d': ax2.plot([])[0], 'data': []})
    graphs.append({'line2d': ax2.plot([])[0], 'data': []})
    graphs.append({'line2d': ax2.plot([])[0], 'data': []})

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
    ax1.set_xlim(0, iterations)
    ax1.set_ylim(0., 3.)

    ax2.legend(handles=[graphs[1]['line2d'], graphs[2]['line2d'], graphs[3]['line2d'], graphs[4]['line2d']], loc='best')
    ax2.grid()
    ax2.set_xlim(0, epochs)
    ax2.set_xticks(range(0, epochs+1))
    ax2.set_ylim(0., 1.)
    ax2.set_yticks(np.arange(0., 1.1, 0.1))

    return fig,


def update_graph(graph_data):
    lines = []
    for idx, data in enumerate(graph_data):
        if data is not None:
            graphs[idx]['data'].append(data)

        line2d = graphs[idx]['line2d']
        line2d.set_data(range(0, len(graphs[idx]['data'])), graphs[idx]['data'])

        lines.append(line2d)

    return lines


def model_frame(input_size, output_size, hidden_sizes=()):
    network.initialize(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)


def model_fit():
    learning_rate = 0.1
    elapsed = 0
    epoch_idx = 0

    for idx in range(1, model_fit.iterations+1):
        # 4-0.
        graph_data = [None, None, None, None, None]

        # 4-1. stopwatch: start
        stime = time.time()

        # 4-2. fetch mini-batch
        batch_mask = np.random.choice(model_fit.train_size, model_fit.batch_size)
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
        if idx % model_fit.epoch_size == 0:
            epoch_idx += 1

            graph_data[1] = network.loss(train_x, train_t)
            graph_data[2] = network.accuracy(train_x, train_t)
            graph_data[3] = network.loss(test_x, test_t)
            graph_data[4] = network.accuracy(test_x, test_t)

            print(f'\nEpoch {epoch_idx:02d}/{model_fit.epochs:02d}')
            print(f'{int(idx/epoch_idx)}/{model_fit.epoch_size} - {elapsed:.3f}s - loss:{graph_data[3]:.4f} - accuracy:{graph_data[4]:.4f}')

            elapsed = 0

        graph_data[0] = graph_data[3] or network.loss(test_x, test_t)

        yield graph_data


if __name__ == '__main__':

    # 1. load training/test data
    (train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    # 2. hyperparamters for model fitting
    model_fit.epochs = 20
    model_fit.batch_size = 100
    model_fit.train_size = train_x.shape[0]
    model_fit.epoch_size = int(model_fit.train_size / model_fit.batch_size)
    model_fit.iterations = model_fit.epochs * model_fit.epoch_size

    # 4. graph set-up
    f, = graph_setup(model_fit.iterations, model_fit.epochs)

    # 5. model frame
    model_frame(train_x.shape[1], train_t.shape[1], hidden_sizes=(50, 100))

    # 6. Realtime Graph of Model Fitting
    ani = animation.FuncAnimation(f, update_graph, model_fit, interval=1, blit=True, save_count=0)
    plt.show()
