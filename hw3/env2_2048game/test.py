import torch

a = torch.arange(8)

print(torch.reshape(a, (2, -1)).shape)