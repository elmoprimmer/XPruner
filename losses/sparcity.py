import torch
import torch.nn as nn
from torch.nn import functional as F

def l2_sparsity(masks):
    reg = 0.0
    for M in masks:
        reg = reg + (M ** 2).sum()
    return reg