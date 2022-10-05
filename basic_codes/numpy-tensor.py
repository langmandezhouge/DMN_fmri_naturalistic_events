
import numpy as np
import torch

x = np.ones((2,2))
print(x)
print(type(x))

x = torch.tensor(x)
#x = torch.from_numpy(x)
print(x)

# tensor-npy
x_= x.numpy()
#x_= x.detach().numpy()
print(x_)
print(type(x_))
