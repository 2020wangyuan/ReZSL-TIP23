import torch

# 示例列表和Tensor
value_list = [7, ]
tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9,7,7,7,7,7])
#tensor = tensor.unsqueeze(0)

# 将列表转换为PyTorch张量
values_tensor = torch.tensor(value_list)
values_tensor= values_tensor.unsqueeze(1)

# 使用torch.nonzero()函数查找满足条件的元素的索引
indices = torch.nonzero(torch.eq(tensor, values_tensor))
print(indices)
