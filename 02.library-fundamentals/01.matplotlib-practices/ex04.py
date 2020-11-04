# figure와 subplot

"""
    1. 크기가 1 X 2 인 Figure 2개의 서브플롯을 추가한 예
"""

from matplotlib import pyplot as plt

fig = plt.figure()

splt1 = fig.add_subplot(1, 2, 1)
splt1.plot([2, 4, 5, 6], [81, 93, 91, 97])

splt2 = fig.add_subplot(1, 2, 2)
splt2.plot([2, 4, 5, 6], [81, 93, 91, 97])

plt.show()
