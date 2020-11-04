# 최소제곱법(Method of Least Squares)

"""
    y = ax + b

"""
import numpy as np

from matplotlib import pyplot as plt

# data
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# 기울기 a, y절편 b 구하기
mx = np.mean(x)
my = np.mean(y)

a = sum([(i - mx)*(j - my) for i, j in tuple(zip(x, y))]) / sum([(i - mx)**2 for i in x])
b = my - (mx * a)

# 결과
print(f'직선 y = {a}x + {b}')
y2 = [(a * i) + b for i in x]

fig, subplots = plt.subplots()
subplots.scatter(x, y)
subplots.plot(x, y2, 'ro-')
plt.show()


