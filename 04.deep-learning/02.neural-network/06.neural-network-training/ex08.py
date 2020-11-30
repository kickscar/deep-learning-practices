# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
# Estimation: Training Accuracy
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

#
# Training Accuracy 와 Testing Accuracy 가 비슷한 것은 overfitting이 일어 나지 않는 것이다.
# 마지막 파라미터로 하지 말고...
# 학습 중에 epoch 별로 Training Accuracy와 Testing Accuracy 를 기록하여 추이를 비교할 것
# 두 Accuracy가 일치해야 하며 차이가 일어 나는 시점에서 학습을 중단한다 - Early Stopping
# Early Stopping, Dropout, Weight Decacy => Prevntation of Overfitting
#
