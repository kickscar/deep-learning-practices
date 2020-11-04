# 색상, 마커, 선 스타일

"""
    plot 메서드의 drawStyle 인자를 통해 점의 연결을 계단모양으로 바꿀수 있다.

    다음과 같은 스타일을 더 적용할 수 있다.
    'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots(1, 1)
subplots.plot(randn(50).cumsum(), drawStyle='steps-mid', color='blue', lineStyle='-', marker='None')

plt.show()
