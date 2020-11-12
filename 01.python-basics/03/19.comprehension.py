results = []
for num in [1, 2, 3, 4, 5, 6, 7, 8]:
    result = num * num
    results.append(result)

print(results)

results = [num*num for num in [1, 2, 3, 4, 5, 6, 7, 8]]
print(results)

# 문자열 리스트에서 길이가 2 이하인 문자열 리스트 만들기
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
strings = [s for s in strings if len(s) <= 2]
print(strings)

# 1~100 사이의 수중에 짝수 리스트 만들 기
evens = [i for i in range(1, 101) if i % 2 == 0]
print(evens)

# [실습] 1~100 사이에 3, 6, 9 가 있는 수 리스트 만들기
# '13'
results = [number for number in range(1, 101) if str(number).count('3') > 0 or str(number).count('6') > 0 or str(number).count('9') > 0]
print(results)

# 문자열 리스트에서 문자열 길이를 순차 자료형으로 저장해 보자
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
lens = [len(s) for s in strings]
print(lens)

# set comprehension
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
strings = {s for s in strings if len(s) <= 2}
print(strings)

# dict comprehension
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
dict = {s: len(s) for s in strings}
print(dict)


