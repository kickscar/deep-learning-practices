# 객체의 대소비교
print(1 > 3)
print(2 < 4)

print(1 >= 3)
print(2 <= 4)

print(1 == 3)
print(2 != 4)

# 복합 관계식 지원
a = 6
print(0 < a and a < 10)
print(0 < a < 10)

# 수치형 이외의 다른 타입의 객체 비교
print('abcd' > 'abc')
print((1, 2, 3) > (1, 2, 2))
print([1, 2, 3] > [1, 2, 2])

# 동질성 비교 :  ==
# 동일성 비교 :  is
a = 20
b = 20
c = a

print(a == b)
print(a is b)
print(a is c)
print(a == c)

# 논리식의 계산순서
print(True or 'logical')
print(False or 'logical')
print([] or 'logical')
print([10, 20] or 'logical')

print('operator' or 'logical')
print('' or 'logical')

print(None or 'logical')
print(None or [])

s = 'Hello World'
s and print(s)

s = ''
s and print(s)

