import torch
import tqdm
import numpy as np

a=torch.randn((3,4,4))
print(a.shape)
print(a.dtype)
a=a.reshape(-1,3)
print(a.shape)