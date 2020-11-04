# while 반복문
count = 1
while count < 11:
    print(count, end=' ')
    count += 1
else:
    print('')

# 1 ~ 10 까지 합
s, i = 0, 1
while i <= 10:
    i += 1
    s += i

print('합: {0}'.format(s))

# break, continue, else 문 적용
i = 0
while i < 10:
    i += 1

    if i < 5:
        continue

    print(i, end=' ')

    if i > 10:
        break
else:
    print('else block')

# 무한루프
i = 0
while True:
    print(i, end=' ')
    if i > 5:
        break
    i += 1
else:
    print('else block')
