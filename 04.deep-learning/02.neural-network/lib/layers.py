import sys
import os
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error
except ImportError:
    raise ImportError("Library Module Can Not Found")


# ReLU Layer
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0.
        return out

    def backward(self, dout):
        dout[self.mask] = 0.
        dx = dout
        return dx


# Affine Layer
class Affine:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b

        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


# Softman with Loss(Cross Entropy) Layer
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


#####################################################


# Multiply Layer
class Multiply:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = self.x * self.y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# Add Layer
class Add:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = self.x + self.y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
