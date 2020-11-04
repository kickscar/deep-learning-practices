# figure와 subplot

"""
    1. 크기가 2 X 1 인 Figure 2개의 서브플롯을 추가한 예
"""

from matplotlib import pyplot as plt

fig = plt.figure()

splt1 = fig.add_subplot(2, 1, 1)
splt1.plot([2, 4, 5, 6], [81, 93, 91, 97])

splt2 = fig.add_subplot(2, 1, 2)
splt2.plot([2, 4, 5, 6], [81, 93, 91, 97])

plt.show()
