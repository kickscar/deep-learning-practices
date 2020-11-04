# figure와 subplot

"""
    하나의 서브플롯에 2개 이상의 그래프를 그리는 것도 가능하다.
    2개이상의 그래프를 구분하기 위해 legend(범례)를 사용할 수 있다.
"""

from matplotlib import pyplot as plt

fig, subplots = plt.subplots(1, 1)
subplots.plot([10, 20, 30, 40], label='data1')
subplots.plot([9, 15, 21, 27], label='data2')

plt.legend(loc='best')
plt.show()
