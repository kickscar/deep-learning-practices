# coding: utf-8
# 다중 선형 회귀(Multiple Linear Regression)
"""
    y = a1x1 + a2x2 + b

"""
import numpy as np
from matplotlib import pyplot as plt

# data
from mpl_toolkits import mplot3d

x1 = [2, 4, 6, 8]
x2 = [0, 4, 2, 3]
y = [81, 93, 91, 97]

# a1, a2, b 초기값
a1 = 0
a2 = 0
b = 0

# epoch
epochs = 60000

# 학습률
learning_rate = 0.001

for i in range(epochs):

    # 편미분값 구하기
    n_data = len(x1)

    arr_x1 = np.array(x1)
    arr_x2 = np.array(x2)
    arr_y = np.array(y)

    arr_y_predic = a1 * arr_x1 + a2 * arr_x2 + b
    arr_error = arr_y - arr_y_predic

    diff_a1 = -(2 / n_data) * sum(arr_error * arr_x1)
    diff_a2 = -(2 / n_data) * sum(arr_error * arr_x2)
    diff_b = -(2 / n_data) * sum(arr_error)

    # 결과출력
    print(f'epoch={i+1}, a1={a1}, diff_a1={diff_a1}, a2={a2}, diff_a2={diff_a2}, b={b}, diff_b={diff_b}')

    # 보정(fitting)
    a1 = a1 - learning_rate * diff_a1
    a2 = a2 - learning_rate * diff_a2
    b = b - learning_rate * diff_b


# 3D 그래프
axes = plt.axes(projection='3d')
axes.scatter(x1, x2, y)
axes.set_xlabel('Study Hours')
axes.set_ylabel('Private Class Hours')
axes.set_zlabel('Score')

plt.show()

# predict!!
x1_p = 0
x2_p = 9
y_p = a1 * x1_p + a2 * x2_p + b
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을 때, 받을 수 있는 점수: {int(y_p)}점')
