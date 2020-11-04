# figure와 subplot

"""
    subplots() 함수 사용

    추가 옵션 사용해 보기
    sharex : 서브플롯이 x축 눈금을 함께 쓴다.
    sharey : 서브플롯이 y축 눈금을 함께 쓴다.
"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots(2, 2, sharex=True, sharey=True)

for i in range(2):
    for j in range(2):
        subplots[i, j].hist(randn(100), bins=20, color='k', alpha=0.3)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)

plt.show()
