# 경사하강법(Gradient Decent)

"""
     y = ax + b
"""
import numpy as np
from matplotlib import pyplot as plt

# data
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# a, b 초기값
a = 0
b = 0

# epoch
epochs = 20000

# 학습률
learning_rate = 0.001

for i in range(epochs):

    # 편미분값 구하기
    n_data = len(x)

    arr_x = np.array(x)
    arr_y = np.array(y)

    arr_y_predic = a * arr_x + b
    arr_error = arr_y - arr_y_predic

    diff_a = -(2 / n_data) * sum(arr_error * arr_x)
    diff_b = -(2 / n_data) * sum(arr_error)

    # 결과출력
    print(f'epoch={i+1}, a={a}, diff_a={diff_a}, b={b}, diff_b={diff_b}')

    # 보정(fitting)
    a = a - learning_rate * diff_a
    b = b - learning_rate * diff_b

