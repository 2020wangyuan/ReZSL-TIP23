import torch

a = torch.tensor([[[1]],[[2]]])
a = a.expand(2,10,1)

print(a.shape)