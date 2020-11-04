# 색상, 마커, 선 스타일

"""
    plot 메서드에 문자열 인자 전달은 의미전달과 복잡해 보이기 때문에 잘 사용하지 않고 다음과 같이 명시적 방법을 선호한다.

    color
    복합 문자열 전달:     k, r,    b,    g,   y,      ....
    명시적 표현 전달: balck, red, blue, gree, yellow, ....
    cf) #rrggbb도 가능

    linestyle : - (solid), - -(dashed), -.(dashdot), dotted, ‘ ‘(None)
    marker : .(dot) v(화살표), o(big dot)
"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots(1, 1)
subplots.plot(randn(50).cumsum(), color='blue', lineStyle=' ', marker='o')

plt.show()
