# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Evaluation from Model Saved
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
model_file = os.path.join(os.getcwd(), 'model', 'model.pkl')
params = None
with open(model_file, 'rb') as f:
    params = pickle.load(f)

# 3. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50, 100], output_size=output_size, init_params=params)

# 4. evaluation
accuracy = network.accuracy(train_x, train_t)
val_accuracy = network.accuracy(test_x, test_t)

print(f'Accuracy: (Train, Test)=({accuracy:.4f}, {val_accuracy:.4f})')
