# figure와 subplot

"""
    subplots_adjust() 함수를 사용하여 서브플롯 간에 적당한 간격과 여백 조정


"""

from matplotlib import pyplot as plt
from numpy.random import randn

fig, subplots = plt.subplots(2, 2, sharex=True, sharey=True)

for i in range(2):
    for j in range(2):
        subplots[i, j].hist(randn(100), bins=20, color='k', alpha=0.3)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

plt.show()