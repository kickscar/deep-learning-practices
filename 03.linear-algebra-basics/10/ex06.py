# coding: utf-8
# 다중선형회귀(해석미분)
import numpy as np


def analytic_gradient(x, data_in, data_out):
    return np.array([
        -2 * np.mean((data_out - (x[:-1] @ data_in + x[-1:])) * (np.array(1) if i is (x.size - 1) else data_in[i]))
        for i in range(x.size)
    ])


def gradient_descent(x, lr=0.01, epoch=100, data_in=None, data_out=None):
    for i in range(epoch):
        gradient = analytic_gradient(x, data_in, data_out)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x


# data
times = np.array([2, 4, 6, 8])
ptimes = np.array([0, 4, 2, 3])
scores = np.array([81, 93, 91, 97])

# 경사하강법
params = gradient_descent(np.array([0., 0., 0.]), epoch=3000, data_in=np.array([times, ptimes]), data_out=scores)

# predict!!
x1_p = 2
x2_p = 2
y_p = params[0] * x1_p + params[1] * x2_p + params[2]
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을 때, 받을 수 있는 점수: {int(y_p)}점')
