# 함수 s를 만드세요. 이 함수는 임의의 개수의 인수를 받아서 그 합을 계산합니다.


def s(*arg):
    return sum(arg, 0)

print(s())
print(s(1, 2))
print(s(1, 2, 5, 7, 2, 3))