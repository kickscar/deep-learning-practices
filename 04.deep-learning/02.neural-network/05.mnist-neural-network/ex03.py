# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification): 신호전달II
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

# 1. 매개변수(w, b 행렬 데이터셋 가져오기)
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 2. 학습/시험 데이터 가져오기
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 3. 신호 전달
count_images = len(train_images)
for index in range(count_images):
    x = train_images[index]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    predict = np.argmax(y)
    label = train_labels[index]

    print(f'image index:{index+1} max:{np.max(y)}, predict:{predict}, label:{label}')
