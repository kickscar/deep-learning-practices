# coding: utf-8
# 신경망 학습: 교차 엔트로피 손실함수 (cross entropy loss function)
import numpy as np


def cross_entropy_error(y, t):
    delta = 1.e-7
    return -np.sum(t * np.log(y + delta))


# support only for one-hot & non-batch
def sum_squares_error(y, t):
    e = 0.5 * np.sum((y - t) ** 2)
    return e


# data
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y1 = [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.]
y2 = [0.1, 0.05, 0.05, 0.6, 0.02, 0.03, 0.05, 0.1, 0., 0.]
y3 = [0., 0., 0.95, 0.02, 0.01, 0.01, 0., 0.01, 0., 0.]

# print(np.sum(y1))
# print(np.sum(y2))
# print(np.sum(y3))

# test1
print(cross_entropy_error(np.array(y1), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))
print(cross_entropy_error(np.array(y3), np.array(t)))

# test2
# cf SSE(Sum Squares Error)
print(sum_squares_error(np.array(y1), np.array(t)))
print(sum_squares_error(np.array(y2), np.array(t)))
print(sum_squares_error(np.array(y3), np.array(t)))
