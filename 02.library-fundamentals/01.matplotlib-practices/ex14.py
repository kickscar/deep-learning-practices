# 제목, 축 이름, 눈금, 눈금 이름

"""
    set_xticks() 메서드를 사용하여 x축의 눈금을 변경할수 있다.

"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots(2, 1)

subplots[0].plot(randn(1000).cumsum())
subplots[1].plot(randn(1000).cumsum())
subplots[1].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

plt.show()
