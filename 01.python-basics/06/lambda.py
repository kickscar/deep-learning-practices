# lambda 함수


def blahblah(x):
    return x * 2


for i in range(10):
    print('{0}:{1}'.format(i, blahblah(i)), end=' ')
else:
    print()


for i in range(10):
    print('{0}:{1}'.format(i, (lambda x: x*2)(i)), end=' ')
else:
    print()


#def strlen(s):
#    return len(s)


strings = ['foo', 'card', 'bar', 'abab', 'aaaa', 'abab', 'foo']
strings.sort(key=lambda s: len(s))
print(strings)

strings.sort(key=lambda s: strings.count(s))
print(strings)



