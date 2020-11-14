# scope test


def func(a):
    return a * x


def func2(a):
    x = 2
    return a * x


def func3(a):
    global g
    g = 3
    return a * g


x = 1
g = 1
print(func(10))
print(func2(10))
print(func3(10))
print(g)

# 내장 영역 (Built-In Scope)
print(dir())
print(dir('__builtins__'))





