# 생성
d = {'basketball': 5, 'baseball': 9}
print(d, type(d))

d2 = dict()
print(d2, type(d2))

d3 = dict(one=1, two=2, three=3, five=5)
print(d3, type(d3))

d4 = dict([('one', 1), ('two', 2), ('three', 3), ('five', 5)])
print(d4, type(d4))


# 인덱스 대신에 key로 접근한다.
print(d['baseball'])

# 연결을 지원하지 않는다.(예외발생)
# d2 = d + {"valleyball": 6}

# 반복(*) 지원하지 않는다.(예외발생)
# d2 = d * 2

# 크기
print(len(d))

# in, not in : 키만 가능
print('soccer' not in d)
print('baseball' in d)

# 다양한 타입의 키를 사용할 수 있다.
d = {}
print(d)

d['twenty'] = 20
d[True] = 'true'
d[10] = 10
print(d)

# 키는 변경불가능한 타입의 값을 사용해야 한다.
# d[[1, 2, 3]] = 6

#
# 객체함수
#
k = d.keys()
print(k, type(k))
for key in k:
    print(key, d[key])

v = d.values()
print(v, type(v))

items = d.items()
print(items)
for t in items:
    print(t)

phones = {'둘리': '0000-0000-0000', '도우넛': '1111-1111-1111', '또치': '2222-2222-2222'}
p = phones
print(phones)
print(p)
phones['마이콜'] = '3333-3333-3333'
print(phones)
print(p)

p = phones.copy()
print(phones)
print(p)
phones['마이콜'] = '4444-4444-4444'
print(phones)
print(p)

print(p.get('마이콜'))
# get() 을 사용하는 이유: 없는 경우에는 None
# []가져오는 경우에는 없을 시 예외
# print(p['길동'])
print(p.get('길동'))

# setDefault : get()과의 차이점은 실제로 저장이 된다.
print(p)
print(p.setdefault('길동', '5555-5555-5555'))
print(p)

# pop() : 삭제와 동시에 값을 가져온다.
phone = p.pop('둘리')
print(phone)
print(p)

t = p.popitem()
print(t)
print(p)

# 모두 삭제
p.clear()
print(p)

# 조회
d = {'c': 3, 'a': 1, 'b': 2}

for key in d:
    print(key, end=' ')
else:
    print('')

for key in d.keys():
    print(key, end=' ')
else:
    print('')

for key, value in d.items():
    print(key, value)














