# coding: utf-8
# XOR Problem

from matplotlib import pyplot as plt

fig, subplots = plt.subplots(1, 2)

subplots[0].scatter([0], [0], marker='o')
subplots[0].scatter([1, 0, 1], [0, 1, 1], marker='^')


subplots[1].scatter([1, 0], [0, 1], marker='o')
subplots[1].scatter([0, 1], [0, 1], marker='^')

plt.show()
