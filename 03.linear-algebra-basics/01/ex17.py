# coding: utf-8
# 행렬 산술연산 - 덧셈

import numpy as np

m1 = np.array([[10, 20, 30], [40, 50, 60]])
print(m1)

# 0 차원 텐서 스칼라는 2차원 행렬로 변환(브로드캐스팅) 후, 요소별 사칙연산을 수행한다.
m2 = m1 + 10
print(m2)

m3 = m1 - 10
print(m3)

m4 = m1 * 10
print(m4)

m5 = m1 / 10
print(m5)

# add(), subtract(), multiply(), divide() 함수도 마찬가지다.
m2 = np.add(m1, 10)
print(m2)

m3 = np.subtract(m1, 10)
print(m3)

m4 = np.multiply(m1, 10)
print(m4)

m5 = np.divide(m1, 10)
print(m5)




