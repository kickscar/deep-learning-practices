# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification) 추론(예측)하기
import pickle
import sys
import os
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    from lib.mnist import load_mnist
except ImportError:
    raise ImportError("lib.mnist Module Can Not Found")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_func(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 대책
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def init_network():
    datasetdir = os.path.join(os.getcwd(), 'dataset')
    with open(datasetdir + '/sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1.T, x) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(W2.T, z1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(W3.T, z2) + b3
    y = softmax(a3)

    return y


# 1. 매개변수(w, b  행렬 데이터셋 가져오기)
network = init_network()

# 2. 학습/시험 데이터 가져오기
(x_train, l_train), (x_test, l_test) = load_mnist(flatten=True, normalize=True)

# 3. 추론(예축) 하기
test_count = len(x_test)
hit = 0
for i in range(test_count):
    x_data = x_test[i]
    label = l_test[i]

    y_data = predict(network, x_data)
    index = np.argmax(y_data)

    if index == label:
        hit += 1

    print(f'test data:{i+1} max:{np.max(y_data)}, index:{index}, label:{label}, hit:{hit}')

# 4. 정확도(Accuracy)
print(f'Accuracy: {hit/test_count}')
