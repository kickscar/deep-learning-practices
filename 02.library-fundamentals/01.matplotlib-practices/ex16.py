# 제목, 축 이름, 눈금, 눈금 이름

"""
    축의 이름: set_xlabel(), set_ylabel() 함수
    그래프 이름: set_title() 함수

"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots()


subplots.plot(randn(1000).cumsum())
subplots.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
subplots.set_xticklabels(['pt0', 'pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'pt9', 'pt10'],
                         rotation=30,
                         fontsize='small')
subplots.set_xlabel('Points')
subplots.set_title('My First Matplotlib Plot')

plt.show()
