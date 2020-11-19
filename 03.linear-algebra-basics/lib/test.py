import numpy as np

data = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1]
])

x = np.array([5, 10, 15])

#x = np.array([[5], [10]])
#print(x[:-1, np.newaxis] * data + x[-1:, np.newaxis])

print(x[:-1] @ data)
print(x[:-1] @ data + x[-1:])

#e = np.mean((x[0] * data_x + x[1] - data_y)**2)

