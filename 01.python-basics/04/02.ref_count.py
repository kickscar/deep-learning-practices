# 레퍼런스 카운트

import sys

x = object()
print(type(x))
print(sys.getrefcount(x))

y = x
print(sys.getrefcount(x))

# 레퍼런스 값 줄이기
del x
print(sys.getrefcount(y))

# symbol x는 심볼 테이블에 사라진다.
# print(x)
