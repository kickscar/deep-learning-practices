# y = ax + b 일차 방정식 결정하기

from matplotlib import pyplot as plt

# 데이터
data_x = [2, 6]
data_y = [81, 91]

# 기울기 a, y절편 b 구하기
a = (data_y[1] - data_y[0]) / (data_x[1] - data_x[0])
b = data_y[1] - a * data_x[1]

# 결과
print(f'직선 y = {a}x + {b}')
y2 = [(a * i) + b for i in data_x]

fig, subplots = plt.subplots()
subplots.plot(data_x, y2, 'ro-')

# cf
# x1 = [2, 4, 6, 8]
# y2 = [81, 93, 91, 97]
# subplots.scatter(x1, y2)

plt.show()
