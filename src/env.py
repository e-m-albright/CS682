"""
Environmental configurations
"""
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
    dtpye = torch.cuda.float16
else:
    device = torch.device('cpu')
    dtype = torch.float16


print('using device:', device)
print('using dtype:', dtype)
