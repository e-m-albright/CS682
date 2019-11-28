"""
Environmental configurations
"""
import torch

from src.args import iargs


dtype = torch.float32


if not iargs.use_cpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


print('Using device:', device)
print('Using dtype:', dtype)
