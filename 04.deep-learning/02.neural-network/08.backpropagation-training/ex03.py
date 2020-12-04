# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Test: Backpropagation Gradient VS Numerical Gradient
import sys
import os
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

# 2. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50], output_size=output_size)

# 3. batched by 3
train_x_batch = train_x[:3]
train_t_batch = train_t[:3]

# 4. gradient
gradient_numerical = network.numerical_gradient(train_x_batch, train_t_batch)
gradient_backpropagation = network.backpropagation_gradient(train_x_batch, train_t_batch)

# 5. means of modulus
for key in gradient_numerical:
    diff = np.average(np.abs(gradient_numerical[key] - gradient_backpropagation[key]))
    print(f'{key} difference: {diff}')

# 6. conclusion - not a little but little!
# w1 : 3.832578941435512e-10
# b1 : 2.231839209735449e-09
# w2 : 5.2156069859722235e-09
# b2 : 1.4029586809238826e-07
