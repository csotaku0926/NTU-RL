import numpy as np

a = np.ones(4) * 100
a[0] = 0.01
b = a < 0.1
print(np.sum(b))