# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification) 배치처리
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
    raise ImportError("lib.mnist Module Can Not Found")


def init_network():
    datasetdir = os.path.join(os.getcwd(), 'dataset')
    with open(datasetdir + '/sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 1. 매개변수(w, b  행렬 데이터셋 가져오기)
network = init_network()

# 2. 학습/시험 데이터 가져오기
(x_train, l_train), (x_test, l_test) = load_mnist(flatten=True, normalize=True)

# 3. 추론(예축) 하기
batch_size = 100
test_count = len(x_test)
hit = 0
for i in range(0, test_count, batch_size):
    x_data = x_test[i:i+batch_size]
    label = l_test[i:i+batch_size]

    # print(x_data.shape)
    y_data = predict(network, x_data)
    index = np.argmax(y_data, axis=1)
    hit += np.sum(index == label)

    # print(f'test data:{i+1} max:{np.max(y_data)}, index:{index}, label:{label}, hit:{hit}')

# 4. 정확도(Accuracy)
print(f'Accuracy: {hit/test_count}')
