# coding: utf-8
# 출력함수(출력층 활성함수) 𝜎() – 소프트맥스함수(Softmax Function)
import numpy as np


# def softmax_func(x):
#     exp_x = np.exp(x)
#     return exp_x / np.sum(exp_x)


def softmax_func(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 대책
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


# test1
a = np.array([0.3, 1., 0.78])
o = softmax_func(a)
print(o)

# test2: 큰값(800.)
# a = np.array([0.3, 800., 0.78])
# o = softmax_func0(a)
# print(o)

# test3: 큰값(800.)
a = np.array([0.3, 800., 0.78])
o = softmax_func(a)
print(o)
