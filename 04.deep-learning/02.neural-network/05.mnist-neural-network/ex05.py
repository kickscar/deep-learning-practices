# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification): 배치시험
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

# 1. 매개변수(w, b  행렬 데이터셋 가져오기)
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 3. 배치시험
hit = 0
xlen = len(test_x)
batch_sz = 100

for idx, sidx_batch in enumerate(range(0, xlen, batch_sz)):
    batch_x = test_x[sidx_batch:sidx_batch+batch_sz]
    # print(batch_x.shape)

    a1 = np.dot(batch_x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    batch_y = softmax(a3)
    # print(batch_y.shape)

    batch_predict = np.argmax(batch_y, axis=1)
    # print(batch_y.shape)

    batch_t = test_t[sidx_batch:sidx_batch+batch_sz]
    # print(batch_t.shape)

    # print(predict_batch == labels_batch)
    batch_hit = np.sum(batch_predict == batch_t)
    hit += batch_hit

    print(f'batch #{idx+1}, batch hit:{batch_hit}, total hit:{hit}')

# 4. 정확도(Accuracy)
print(f'Accuracy: {hit/xlen}')
