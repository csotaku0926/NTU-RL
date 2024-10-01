from collections import deque
import numpy as np

# assume 2 states, 3 actions
rng = np.random.default_rng(seed=1)
idx = rng.choice(5, 3, replace=False)

print(idx)