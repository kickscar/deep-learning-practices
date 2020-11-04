# zip() 함수 사용 예

seq1 = {'foo', 'bar', 'baz'}
seq2 = {'one', 'two', 'three'}

z = zip(seq1, seq2)
print(z)
print(type(z))

# 순회
for t in z:
    print(t, type(t))

z = zip(seq1, seq2)
for index, t in enumerate(z):
    print(index, t)


z = zip(seq1, seq2)
for index, (a, b) in enumerate(z):
    print(index, a, b)


d1 = [('foo', 'one'), ('bar', 'two'), ('baz', 'three')]
seq1, seq2 = zip(*d1)
print(seq1)
print(seq2)

