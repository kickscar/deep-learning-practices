# global, local 심볼 테이블 내용 확인
def f():
    print(g_a)
    l_a = 2
    l_b = '마이콜'
    print(locals())


class MyClass:
    x = 10
    y = 20


g_a = 1
g_b = '둘리'

print(globals())
f()


# 1. 정의된 함수 객체
f.k = 'hello'
print(f.__dict__)


# 2. 클래스 객체
print(MyClass.__dict__)  # 사용자 정의 클래스 객체
print(int.__dict__)      # 내장 정의 클래스 객체


# 3. 인스탄스 객체
o = MyClass()
o.x = 100
print(o.__dict__)


# 내장 함수 또는 내장 클래스의 인스탄스 객체 __dict__ 가 없다.
# 즉, 심볼 테이블이 없으므로 확장이 불가능하다.
# print(print.__dict__)
# print(g_a.__dict__)