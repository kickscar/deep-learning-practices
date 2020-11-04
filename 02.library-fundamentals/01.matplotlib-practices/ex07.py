# figure와 subplot

"""
    subplots() 함수 사용
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy.random import randn

fig, splts = plt.subplots(2, 2)

splts[0, 0].plot([2, 4, 5, 6], [81, 93, 91, 97])
splts[0, 1].plot(randn(50).cumsum(), 'k--')
splts[1, 0].hist(randn(100), bins=20, color='k', alpha=0.3)
splts[1, 1].scatter(np.arange(100), np.arange(100) + 3 * randn(100))

plt.show()
