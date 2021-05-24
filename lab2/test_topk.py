import numpy as np
a = np.array([1, 30, 2, 4, 15])
ind = np.argpartition(a, -3)[-3:]
idx = np.argwhere(a > 10).ravel()
print(a[idx])
