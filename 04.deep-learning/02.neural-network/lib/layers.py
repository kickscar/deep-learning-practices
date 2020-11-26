import pickle
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
        self.mask = (x <= 0)  # 매개변수로 들어온 넘파이배열 x의 원소가  0이하인지 판단하기
        out = x.copy()  # 원본 배열 복사 후
        out[self.mask] = 0  # 0보다 작은 원소들을 0으로 만들기

        return out

    def backward(self, dout):
        dout[self.mask] = 0  # 상류에서 들어온 값에서 0보다 작은 값들에 대해 0으로 치환
        dx = dout  # 완성된 ReLU 배열 리턴

        return dx


# Affine Layer
class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b

        self.x = None
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


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


# Multiply Layer
class Multiply:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y
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

        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
