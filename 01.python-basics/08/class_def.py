# 클래스 정의 테스트


class MyString:
    pass


s = MyString()
print(type(s))
print(MyString.__bases__)

s2 = str()
print(type(s2))
print(str.__bases__)
