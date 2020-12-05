# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
# Evaluation from Model Saved
import pickle
import sys
import os
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load params dataset trained
params_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_params.pkl')
with open(params_file, 'rb') as f:
    network.params = pickle.load(f)

accuracy = network.accuracy(train_x, train_t)
print(f'Training Accuracy: {accuracy}')

accuracy = network.accuracy(test_x, test_t)
print(f'Testing Accuracy: {accuracy}')
