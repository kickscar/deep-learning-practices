# coding: utf-8
import numpy as np


# step function(계단 함수)
def step(x):
    return np.array(x > 0, dtype=np.int)


# indentity function(항등 함수)
def identity(x):
    return x
