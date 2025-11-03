import torch
import torch.nn as nn
from torch.nn import functional as F


def l2_sparsity(masks):
    """
    Eq. 7 in paper
    """
    reg = 0.0
    for M in masks:
        reg = reg + (M ** 2).sum()
    return reg

def laplacian_1d(x):
    # Discrete 2nd derivative along last dim: x_{i-1} - 2 x_i + x_{i+1}
    pad_l = F.pad(x, (1,0), mode='constant', value=0)
    print(pad_l.shape)
    pad_r = F.pad(x, (0,1), mode='constant', value=0)
    print(pad_r.shape)
    return pad_l - 2*x + pad_r


def smoothness_laplacian(masks, reduce_over=("feature",)):
    #todo fix
    """
    Eq. 6 in paper
    masks: list of class-wise mask tensors
       attention head mask: [C, H, D]  (classes, heads, dim_per_head)
       linear/FFN mask:     [C, M, N]  (classes, out_dim, in_dim)
    reduce_over: which axes to apply 2nd difference on. Options: "feature", "head", "row", "col".
    We use L1 of 2nd-diff
    """
    reg = 0.0
    for M in masks:
        R = 0.0
        if M.ndim >= 2 and "feature" in reduce_over:
            R = R + laplacian_1d(M).abs().sum()
        if M.ndim >= 3 and "head" in reduce_over:
            R = R + laplacian_1d(M.transpose(-2, -1)).transpose(-2, -1).abs().sum()
        reg = reg + R
    return reg


class XPrunerLoss(nn.Module):
    def __init__(self, lambda_smooth=1e-4, lambda_sparse=1e-6):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, mask_tensors):
        """
        logits: model outputs after applying class-wise masks per sample
        targets: [B]
        mask_tensors: list of all learnable mask tensors across layers/units (class-wise)
        """
        loss_ce = self.ce(logits, targets)
        #L_sm = smoothness_laplacian(mask_tensors, reduce_over=("feature",))
        L_sm = 0.0
        L_sp = l2_sparsity(mask_tensors)
        loss = loss_ce + self.lambda_smooth * L_sm + self.lambda_sparse * L_sp
        return loss, {"ce": loss_ce.detach(), "sparse": L_sp.detach()}



