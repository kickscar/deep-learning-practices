# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Test: Comparings Optimizer
import sys
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
    from optimizer import SGD, Momentum, AdaGrad, Adam
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 01. optimizer set-up
optimizers = {
    'sgd': SGD(),
    'momentum': Momentum(),
    'adagrad': AdaGrad(),
    'adam': Adam(),
}

# 02. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 03. hyperparamters
batch_size = 100
epochs = 30

# 04. testing
train_losses = dict()

for optimizer_key in optimizers.keys():

    train_losses[optimizer_key] = []

    # 05. model frame
    network.initialize(input_size=train_x.shape[1], hidden_size=[100, 100, 100], output_size=train_t.shape[1])

    # 06. model fitting
    train_size = train_x.shape[0]
    epoch_size = int(train_size / batch_size)
    iterations = epochs * epoch_size

    elapsed = 0
    epoch_idx = 0

    for idx in range(1, iterations+1):

        # 06-01. start stopwatch
        stime = time.time()

        # 06-02. fetch mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        train_x_batch = train_x[batch_mask]
        train_t_batch = train_t[batch_mask]

        # 06-03. gradient
        gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)

        # 06-04. update parameters
        for key in network.params:
            # network.params[key] -= learning_rate * gradient[key]
            optimizers[optimizer_key].update(network.params, gradient)

        # 06-05. train loss
        loss = network.loss(train_x_batch, train_t_batch)
        train_losses[optimizer_key].append(loss)

        # 06-06. start stopwatch
        elapsed += (time.time() - stime)

        # 06-07. epoch accuracy
        if idx % epoch_size == 0:
            epoch_idx += 1

            print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')
            print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000:.4f}ms - loss:{loss:.4f}')

            elapsed = 0

    network.params = dict()
    network.layers = []


# 05. graph
def smooth_curve(x):
    """ 그래프를 매끄럽게 하기 위해 사용
        참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[5:len(y) - 5]


markers = {"sgd": "o", "momentum": "x", "adagrad": "s", "adam": "D"}
x = np.arange(iterations)

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_losses[key]), marker=markers[key], markevery=100, label=key)

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()

plt.show()
