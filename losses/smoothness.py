import torch
import torch.nn as nn
from torch.nn import functional as F


def laplacian_1d(x):
    # Discrete 2nd derivative along last dim: x_{i-1} - 2 x_i + x_{i+1}
    pad_l = F.pad(x, (1,0), mode='constant', value=0)
    pad_r = F.pad(x, (0,1), mode='constant', value=0)
    return pad_l - 2*x + pad_r