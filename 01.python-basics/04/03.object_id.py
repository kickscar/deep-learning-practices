import sys

# Object ID

#
#
i1 = 10
i2 = 10
print(hex(id(i1)), hex(id(i2)))

i1 = 12345678901234567890
i2 = 12345678901234567890
print(hex(id(i1)), hex(id(i2)))

#
#
l1 = [1, 2, 3]
l2 = [1, 2, 3]
print(hex(id(l1)), hex(id(l2)))

#
#
s1 = 'hello'
s2 = 'hello'
print(hex(id(s1)), hex(id(s2)))

print(i1 is i2)
print(s1 is s2)
print(l1 is l2)

# ?
t1 = (1, 2, 3)
t2 = (1, 2, 3)
print(sys.getrefcount(t1), sys.getrefcount(t2))
print(t1 is t2)

t3 = t1
print(sys.getrefcount(t1), sys.getrefcount(t2), sys.getrefcount(t3))

t4 = tuple(range(1, 4))
print(sys.getrefcount(t4))


t5 = t1[1:]
print(t5)
print(sys.getrefcount(t5))
