import copy

# 렌퍼런스 복사
a = 1
b = a

a = [1, 2, 3]
b = [4, 5, 6]
x = [a, b, 100]
y = x

print(x)
print(y)
print(x is y)

# swallow copy(얕은 복사)
a = [1, 2, 3]
b = [4, 5, 6]
x = [a, b, 100]
y = copy.copy(x)

print(x)
print(y)
print(x is y)
print(x[0] is y[0])

# deep copy(깊은 복사)
a = [1, 2, 3]
b = [4, 5, 6]
x = [a, b, 100]
y = copy.deepcopy(x)

print(x)
print(y)
print(x is y)
print(x[0] is y[0])

# 깊은 복사가 복합객체만을 생성하기 때문에
# 복합 객체가 한개만 있는 경우,
# 얕은 객체와 깊은 복사는 차이가 없다.
a = ["hello", "world"]
b = copy.copy(a)
print(a is b)
print(a[0] is b[0])

c = copy.deepcopy(a)
print(a is c)
print(a[0] is c[0])
