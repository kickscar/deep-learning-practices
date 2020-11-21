# coding: utf-8
# 3층 신경망 구현하기 – 신호전달 구현1: 은닉1층 전달
import numpy as np

print('\n= 신호전달 구현1: 은닉1층 전달 ==============================')

w1 = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 1.]])
print(f'w1 dimension: {w1.shape}')  # 3 X 2 matrix

x = np.array([1., 5.])
print(f'x dimension: {x.shape}')    # 2 vector

b1 = np.array([0.1, 0.2, 0.3])
print(f'b1 dimension: {b1.shape}')  # 3 vector

# 오류!
# 일차함수(식) 중심으로 생각하지 말고
# a 총합을 내기 위한 것이니깐 w 중심으로 생각 할 것
# a1 = np.dot(x, w1) + b1
a1 = np.dot(w1, x) + b1

# tensor flows~
# 3 x 2(m)  2(v) -> 3(v)
# tensor1(크기2) 입력신호가 뉴런에서 tensor2(가중치)와 총합으로 tensor1(크기3) 출력신호가 되었다.
print(f'a1 = {a1}')
