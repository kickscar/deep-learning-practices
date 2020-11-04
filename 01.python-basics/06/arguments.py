# 기본 인수값


def incr(a, step=1):
    return a + step


print(incr(10, step=2))
print(incr(10, 2))
print(incr(10))

# 오류
# def decr(step=1, a):
#    return a-step


# 키워드 인수
def area(width, height):
    return width*height


print(area(10, 20))
print(area(width=10, height=20))
print(area(height=20, width=10))
print(area(10, height=20))
# 오류
# print(area(10, width=20))
# print(area(height=10, 20))


# 가변 인수
def vargs(a, *arg):
    print(a, arg)

# vargs()
vargs(10)
vargs(10, 1)
vargs(10, 1, 2, 3, 4, 5)

def vargs2(*arg):
    print(arg)

vargs2()
vargs2(10)
vargs2(10, 20, 30)

# 모든 인수를 sum
def sum(*numbers):
    s = 0;
    for n in numbers:
        s += n
    return s

print(sum())
print(sum(1, 2, 3, 4, 5))

# 내장함수 print는 어떻게 만들어졌을까?

def _print(*arg, e='newline'):
    print(arg, e)

_print(10, 20, 30)
_print(10, 20, 30, e='tab')


# c의 printf함수 흉내내기   printf("%s이 %d원짜리 %s입고 %s를 차고 노래를 한다", "타잔", 100, "팬티", "칼")
def printf(f, *arg):
    print(f % arg)

printf("%s이 %d원짜리 %s입고 %s를 차고 노래를 한다", "타잔", 100, "팬티", "칼")


# 정의되지 않은 키워드 인수 처리하기
def f(width, height, **kwd):
    print(width, height, kwd['depth'], kwd['dimension'])

f(10, 20, depth=30, dimension=3)
# 오류
# f(10, 20, depth=30, 3)


def g(a,b, *arg, **kwd):
    print(a, b)
    print(arg)
    print(kwd)

g(10, 20)
g(10, 20, 30)
g(10, 20, c=6)
g(10, 20, 30, 40, 50, c=6, d=7)


def h(name, age, height):
    print(name, age, height)

h('둘리', 10, 140)

t = ('둘리', 10, 140)
h(*t)

d = {"name": "둘리", "age": 10, "height": 140}
h(**d)