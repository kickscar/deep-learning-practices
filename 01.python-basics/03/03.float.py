a = 1.2
print(a, type(a))

# is_integer 함수는 값으로 정수인지 실수인지 판단한다.
b = 2.0
print(type(b))
print(b.is_integer())

# 다른 언어처럼 소수점 표현뿐만 아니라 e, E 지수표현도 가능하다.
c = 3e3
print(c)
d = 0.2e-4
print(d)