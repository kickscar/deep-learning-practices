# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification) 구현하기
import pickle
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import sigmoid, softmax
except ImportError:
    raise ImportError("Library Module Can Not Found")


def init_network():
    datasetdir = os.path.join(os.getcwd(), 'dataset')
    with open(datasetdir + '/sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


# 1. 매개변수(w, b  행렬 데이터셋 가져오기)
network = init_network()

# 2. 학습/시험 데이터 가져오기
(x_train, l_train), (x_test, l_test) = load_mnist(flatten=True, normalize=True)

for i in range(len(x_train)):
    x_data = x_train[i]
    label = l_train[i]
    a1 = np.dot(network['W1'].T, x_data) + network['b1']
    # print(network['W1'].shape)
    # print(x_train.shape)
    # print(network['b1'].shape)
    # print(a1)
    # print('===================================================================')

    z1 = sigmoid(a1)
    # print(z1)
    # print('===================================================================')

    a2 = np.dot(network['W2'].T, z1) + network['b2']
    # print(network['W2'].shape)
    # print(z1.shape)
    # print(network['b2'].shape)
    # print(a2)
    # print('===================================================================')

    z2 = sigmoid(a2)
    # print(z2)
    # print('===================================================================')

    a3 = np.dot(network['W3'].T, z2) + network['b3']
    # print(network['W3'].shape)
    # print(z2.shape)
    # print(network['b3'].shape)
    # print(a3)
    # print('===================================================================')

    y = softmax(a3)
    print(f'image index:{i} max:{np.max(y)}, index:{np.where(np.max(y) == y)}, label:{label}')
    # print('===================================================================')

