# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Estimation: Training Accuracy
import pickle
import sys
import os
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load model
params_file = os.path.join(os.getcwd(), 'model', 'model.pkl')
params = None
with open(params_file, 'rb') as f:
    params = pickle.load(f)

# 3. model frame
network.initialize(input_size=train_x.shape[1], hidden_size=[50], output_size=train_t.shape[1], init_params=params)

accuracy = network.accuracy(train_x, train_t)
print(f'Training Accuracy: {accuracy}')

accuracy = network.accuracy(test_x, test_t)
print(f'Testing Accuracy: {accuracy}')
