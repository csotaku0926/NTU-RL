import numpy as np

a = np.array([
    [4, 2, 2, 0],
    [1, 3, 2, 2],
    [2, 4, 2, 4],
    [1, 1, 2, 4]
])

corner_reward = 1.0 
weight = np.array([
        [corner_reward  , 0  , 0  , corner_reward  ],
        [0  , 0  , 0  , 0  ],
        [0  , 0  , 0  , 0  ],
        [corner_reward  , 0  , 0  , corner_reward  ]])

M = a.flatten()
print(np.dot(M == M.max(), weight.flatten()))