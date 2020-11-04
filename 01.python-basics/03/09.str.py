# 한줄 문자열
s = ''
str1 = 'Hello World'
str2 = "Hello World"

print(type(s), type(str1), type(str2))
print(isinstance(str2, str))

# 여러줄 문자열
str3 = """ABC
DEF
가나다라마바사
아자차카타파하"""
print(str3)

# 여러줄 주석 => 여러줄 문자열을 사용한다.
"""
주석1
주석2
주석3
"""

# escape
print('hello\nworld\n')
print("I Say \"HelloWorld\"")
print('I Say "HelloWorld"')
print('She\'s Mom')
print("She's Mom")
print("둘리\t010-0000-0000")

# 문자열 연산
str1 = "First String"
str2 = "Second String"

# 반복
str4 = str1*3
print(str4)

# 인덱싱
print(str1[0], str1[2], str1[4])

# 슬라이싱(slicing)
str5 = str2[2:5]
print(str5)
print(str2[2:])

# 연결(+), + 생략 가능
str3 = str1 + ' ' + str2
print(str3)
str6 = 'Hello' '-' 'World'
print(str6)

# 문자열 객체와 정수형 객체는 + 연산을 할 수 없다.
name = '길동'
age = 40
# print(name + age)
print(name + str(age))

# len() 함수
print(len(str2))

# in, not in 연산자
print("S" in str2)
print("S" not in str2)

# 문자열 객체는 변경할 수 없다 (immutable)
# str1[0] = "f"
# print(str1)

# formating(서식) - tuple
f = "name => %s, age => %d"
print(f)
print("name => %s, age => %d" % ('둘리', 10))
print(f % ('또치', 20))

# formating(서식) - format() 함수
name = "마이콜"
age = 30
print("name => " + name + ", age => " + str(age))
print("name => " + name + ", age => " + format(age, "d"))
print("name => " + format(name, "s") + ", age => " + format(age, "d"))

# 대소문자 관련 객체함수
s = "i like Python"
print(s.upper())
print(s.lower())
print(s.swapcase())
print(s.capitalize())
print(s.title())

# 검색 관련 객체함수
s = "I Like Python, I Like Java Also"
print(s.count("Like"))
print(s.find("Like"))
print(s.find("Like", 5))
print(s.find("JavaScript"))
print(s.rfind("Like"))

# 발견하지 못하면 예외가 발생한다.
# print(s.index("JavaScript"))
print(s.rindex("Like"))
print(s.startswith('I Like'))
print(s.startswith('I Like', 2))
print(s.endswith('Also'))
print(s.endswith('Java', 0, 26))

# 편집과 치환
s = '      spam and ham      '
print('---' + s.strip() + '---')
print('---' + s.rstrip() + '---')
print('---' + s.lstrip() + '---')

s = '<><abc><><defg><>'
print('---' + s.strip('<>') + '---')

s = 'Hello Java Java Java'
print(s.replace('Java', 'Python'))

# 정렬
s = 'King and Queen'
print('---' + s.center(30) + '---')
print('---' + s.ljust(30) + '---')
print('---' + s.rjust(30) + '---')

# 분리
s = 'spam and ham'
r = s.split(' and ')
print(r, type(r))

s = 'one:two:three:four'
r = s.split(':')
print(r)

r = s.split(':', 2)
print(r)

r = s.rsplit(':', 2)
print(r)

lines = """1st line
2nd line
3rd line
4th line
"""
r = lines.split('\n')
print(r)
r = lines.splitlines()
print(r)

# 결합
s = '&'.join(r)
print(s)

# 판별
print("1234".isdigit())
print("abcd".isalpha())
print("1234".isalpha())
print("abcd".isdigit())
print("abcdef".islower())
print("ABCDEF".isupper())
print(" ".isspace())
print("".isspace())
print("\n".isspace())
print("\t".isspace())

# '0' 채우기
print(str(1).zfill(5))
print(str(9234).zfill(5))

# formating(서식)
print("name:{}, age:{}".format("둘리", 10))
print("name:{0}, age:{1}".format("둘리", 10))
print("name:{1}, age:{0}".format(10, "둘리"))

f = "name:{n}, age:{a}"
s = f.format_map({"n": "둘리", "a": 10})
print(s)



























