# enumerate() 함수 사용

i = 0
for color in ['red', 'yellow', 'blue', 'black', 'gray']:
    print(i, color)
    i = i + 1

for i, color in enumerate(['red', 'yellow', 'blue', 'black', 'gray']):
    print(i, color)
