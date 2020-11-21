# coding: utf-8
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification): 신호전달I
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

# 2. 학습/시험 데이터 가져오기
(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
# count_images = train_images.shape[0]
count_images = len(train_images)
index_random = np.random.randint(0, count_images, 1).reshape(())

# 3. 신호전달
print('\n= 신호전달 구현1: 은닉1층 전달 ==============================')

x = train_images[index_random]
print(f'x dimension: {x.shape}')    # 784 vector
w1 = network['W1']
print(f'w1 dimension: {w1.shape}')  # 784 x 50 matrix
b1 = network['b1']
print(f'b1 dimension: {b1.shape}')  # 50 vector
a1 = np.dot(x, w1) + b1
print(f'a1 = {a1}')

print('\n= 신호전달 구현2: 은닉1층 활성함수 h() 적용 ===================')

print(f'a1 dimension: {a1.shape}')  # 50 vector
z1 = sigmoid(a1)
print(f'z1 = {z1}')

print('\n= 신호전달 구현3: 은닉2층 전달 ==============================')

print(f'z1 dimension: {z1.shape}')  # 50 vector
w2 = network['W2']
print(f'w2 dimension: {w2.shape}')  # 50 X 100 matrix
b2 = network['b2']
print(f'b2 dimension: {b2.shape}')  # 100 vector
a2 = np.dot(z1, w2) + b2
print(f'a2 = {a2}')

print('\n= 신호전달 구현4: 은닉2층 활성함수 h() 적용 ===================')

print(f'a2 dimension: {a2.shape}')  # 100 vector
z2 = sigmoid(a2)
print(f'z2 = {z2}')

print('\n= 신호전달 구현5: 출력층 전달 ===============================')

print(f'z2 dimension: {z2.shape}')  # 100 vector
w3 = network['W3']
print(f'w3 dimension: {w3.shape}')  # 100 X 10 matrix
b3 = network['b3']
print(f'b3 dimension: {b3.shape}')  # 10 vector
a3 = np.dot(z2, w3) + b3
print(f'a3 = {a3}')

print('\n= 신호전달 구현6: 출력층 활성함수 𝜎() 적용 ====================')

print(f'a3 dimension: {a3.shape}')  # 2 vector
y = softmax(a3)
print(f'y = {y}')

print('\n= 예측 결과 ============================================')
calss = np.argmax(y)
print(f'{index_random+1} 번째 이미지 예측: {calss}')

print('\n= 정답 ================================================')
label = train_labels[index_random]
print(f'{index_random+1} 번째 이미지 레이블: {label}')
