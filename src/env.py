"""
Environmental configurations
"""
import torch


dtype = torch.float16


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


print('using device:', device)
print('using dtype:', dtype)
