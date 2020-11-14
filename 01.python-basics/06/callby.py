# call by reference 이지만
# 내부에서 변경하더라도 변경되지 않는다.(immutable)


def f(i):
    i = 20


def f2(seq):
    seq[0] = 0


a = 10
f(a)
print(a)

# 파라피터가 sequence type인 경우에는 내부에서 변경시
# 오류
# a = (1, 2, 3)
# f2(a)

# mutable sequence 타입인 경우 수정이 가능하다.
a = [1, 2, 3]
f2(a)
print(a)
