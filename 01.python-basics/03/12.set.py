# 생성
s = {1, 2, 3}
print(s, type(s))

# 기본연산
print(len(s))
print(2 in s)
print(10 not in s)

# list의 중복성 제거
nums = [1, 2, 3, 4, 2, 2, 5, 5, 6]

s = set(nums)
print(s, type(s))

nums = list(s)
print(nums)

#
# 객체함수
#
s.add(7)
print(s)

s.add(2)
print(s)

s.discard(2)
print(s)

# remove는 discard와 다르게 없는 원소 제거시 예외가 발생
# s.remove(30)
s.remove(3)
print(s)

s.update({6, 7, 8})
print(s)

s.clear()
print(s)

# 수학의 집합과 관련된 객체 함수
s1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
s2 = {10, 20, 30}

s3 = s1.union(s2)
print(s3)

s4 = s1.intersection(s2)
print(s4)

s5 = s1.difference(s2)
print(s5)

s6 = s1.symmetric_difference(s2)
print(s6)


