import numpy as np

y = np.array([0.123, 0.456, 0.1234])

r = np.where(np.max(y) == y)
print(r[0].squeeze(0))