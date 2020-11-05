# coding: utf-8
# 최소제곱법(Method of Least Squares)
"""
    y = ax + b

"""
import numpy as np
from matplotlib import pyplot as plt

# data
data_x = [2, 4, 6, 8]
data_y = [81, 93, 91, 97]

# 기울기 a, y절편 b 구하기
mx = np.mean(data_x)
my = np.mean(data_y)

a = sum([(i - mx)*(j - my) for i, j in tuple(zip(data_x, data_y))]) / sum([(i - mx)**2 for i in data_x])
b = my - (mx * a)

# 결과
print(f'직선 y = {a}x + {b}')
data_y2 = [(a * i) + b for i in data_x]

fig, subplots = plt.subplots()
subplots.scatter(data_x, data_y)
subplots.plot(data_x, data_y2, 'ro-')
plt.show()
