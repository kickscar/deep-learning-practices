# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification): 배치처리
import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
    from common import sigmoid, softmax
except ImportError:
    raise ImportError("Library Module Can Not Found")


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
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 2. 학습/시험 데이터 가져오기
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 3. 추론(예측) 하기
hit = 0
count_images = len(test_images)
batch_size = 100

for index, sindex_batch in enumerate(range(0, count_images, batch_size)):
    x_batch = test_images[sindex_batch:sindex_batch+batch_size]

    # print(x_batch.shape)

    a1 = np.dot(x_batch, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y_batch = softmax(a3)

    predict_batch = np.argmax(y_batch, axis=1)
    labels_batch = test_labels[sindex_batch:sindex_batch+batch_size]

    hit += np.sum(predict_batch == labels_batch)

    print(f'batch #{index+1}, hit:{hit}')

# 4. 정확도(Accuracy)
print(f'Accuracy: {hit/count_images}')
