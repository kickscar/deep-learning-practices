# 함수 정의


def dummy():
    pass


def my_function():
    print("Hello World")


def times(a, b):
    return a * b


def none():
    return


dummy()
my_function()
print(dummy())
print(times(10, 10))
print(none())

# 함수도 객체다
print(dummy, type(dummy))

f = my_function
f()


def my_function2(func):
    func()


my_function2(f)

print(f, my_function, sep=',')

# 여러 return 값
def func():
    return 10, 'hello', {1, 2, 3}, (1, 2, 3)

n, s, s, t = func()
print(n, s, s, t)
