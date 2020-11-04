# 평균제곱오차(MSE, Mean Squares Error)

"""
    y = ax + b

"""
import numpy as np
from matplotlib import pyplot as plt

# data
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]


# a, b 가정
a = 3
b = 75

# 예측 점수
y_predict = [a * i + b for i in x]

# mse 구하기
mse = np.mean([(yp - y)**2 for yp, y in tuple(zip(y_predict, y))])

# 결과출력
for t, p, pp in tuple(zip(x, y, y_predict)):
    print(f'공부한 시간={t}, 실제점수={p}, 예측점수={pp}')

print(f'오차(평균제곱오차):{mse}')

fig, subplots = plt.subplots()
subplots.scatter(x, y)
subplots.plot(x, y_predict, 'ro-')
plt.show()
