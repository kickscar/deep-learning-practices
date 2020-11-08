# coding: utf-8
# Logistic Regression
import numpy as np
from matplotlib import pyplot as plt


# sigmoid 함수
def sigmoid(v):
    return 1 / (1 + np.e ** (-v))


# data
x = [2, 4, 6, 8, 10, 12, 14]
y = [0, 0, 0, 1, 1, 1, 1]


# a, b 값 초기화
a = 0
b = 0

# epoch
# 1 epochs = 10000
epochs = 60000

# 학습률
learning_rate = 0.05

# gradient decent algorithm
for i in range(epochs):
    for data_x, data_y in tuple(zip(x, y)):
        diff_a = data_x * (sigmoid(a * data_x + b) - data_y)
        diff_b = sigmoid(a * data_x + b) - data_y

        # 결과출력
        print(f'epoch={i + 1}, a={a}, diff_a={diff_a}, b={b}, diff_b={diff_b}')

        # 보정
        a = a - learning_rate * diff_a
        b = b - learning_rate * diff_b


# # graph
# fig, subplots = plt.subplots(1, 1)
# subplots.scatter(x, y)
#
# x2 = list(np.arange(0, 15, 0.1))
# subplots.plot(x2, [sigmoid(a * value + b) for value in x2])
#
# plt.show()
#
#
# # predict
# print(sigmoid(a * 7 + b))
