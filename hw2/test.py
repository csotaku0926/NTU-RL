from collections import deque
import numpy as np

# assume 2 states, 3 actions
b = np.array([1,2,3])
c = np.array([4,5,6])
a = [b, c]

print(np.stack(a))