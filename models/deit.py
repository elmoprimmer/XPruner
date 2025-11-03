from data import imagenet as imgnt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import VisionTransformer

import random
import inspect

from PIL import Image
import requests
import matplotlib.pyplot as plt




class MaskedAttn(nn.Module):
    """
    Attn module with mask
    """
    def __init__(self, attn: nn.Module, num_classes: int):
        super().__init__()
        
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop
        self.scale = attn.scale
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.num_classes = num_classes
        self.mask = nn.Parameter(torch.ones(self.num_classes, self.num_heads, self.head_dim)) # (C, H, D)
    
        for p in self.qkv.parameters(): p.requires_grad = False
        for p in self.proj.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        B, N, C = x.shape # B = batch, N = num tokens (cls+patches), C = embed dim (head dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D)

        q = q * self.scale
        a = q @ k.transpose(-2, -1)
        a = a.softmax(dim=-1)
        a = self.attn_drop(a)
        o = a @ v

        if y is not None:
            M = self.mask[y] # (B, H, D)
            M = M.unsqueeze(2) # (B, H, 1, D)
        else:
            assert 1 == 2, "todo"
            
            #todo: should take avg over classes and multiply

        o = o * M

        x = o.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MaskedDeiT(nn.Module):
    """
    deit with masked attn
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes

        self.masked_attn = nn.ModuleList()
        for blk in self.model.blocks:
            self.masked_attn.append(MaskedAttn(blk.attn, num_classes = self.num_classes))

        # turn off original attn modules
        for blk in self.model.blocks:
            for p in blk.attn.parameters(): p.requires_grad = False


    def forward_features(self, x, y=None):
        B = x.shape[0] # batch size
        x = self.model.patch_embed(x)
        cls_tok = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1) 
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        
        for i, blk in enumerate(self.model.blocks):
            attn_out = self.masked_attn[i](blk.norm1(x), y)
            x = x + blk.drop_path1(blk.ls1(attn_out))
            
            mlp_out = blk.mlp(blk.norm2(x))
            x = x + blk.drop_path2(blk.ls2(mlp_out))
            
        
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x, y=None):
        x = self.forward_features(x, y)
        return self.model.head(x)


