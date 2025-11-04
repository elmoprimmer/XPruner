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
        self.mask_attn = nn.Parameter(torch.ones(self.num_classes, self.num_heads, self.head_dim)) # (C, H, D)
        out_proj, in_proj = self.proj.weight.shape
        self.mask_proj = nn.Parameter(torch.ones(self.num_classes, out_proj, in_proj)) # (C, out_proj, in_proj)

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
            M = self.mask_attn[y] # (B, H, D)
            M = M.unsqueeze(2) # (B, H, 1, D)
            M_proj = self.mask_proj[y]
        else:
            assert 1 == 2, "todo"
            #todo: should take avg over classes and multiply

        o = o * M
        x = o.transpose(1, 2).reshape(B, N, C)
        
        W_proj = self.proj.weight.unsqueeze(0)*M_proj # B, out, in
        x = torch.einsum('bni,boi->bno', x, W_proj) + self.proj.bias.unsqueeze(0).unsqueeze(0) # B, N, C
        # einsum bc we have weights with batch dim, so need batch mat mul
        
        x = self.proj_drop(x)
        return x

class MaskedMlp(nn.Module):
    """
    MLP block with mask
    """
    def __init__(self, mlp: nn.Module, num_classes: int):
        super().__init__()

        self.fc1 = mlp.fc1
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.norm = mlp.norm
        self.fc2 = mlp.fc2
        self.drop2 = mlp.drop2

        for p in self.fc1.parameters(): p.requires_grad = False
        for p in self.fc2.parameters(): p.requires_grad = False
        
        self.num_classes = num_classes

        out1, in1 = self.fc1.weight.shape
        out2, in2 = self.fc2.weight.shape

        self.mask_fc1 = nn.Parameter(torch.ones(self.num_classes, out1, in1))
        self.mask_fc2 = nn.Parameter(torch.ones(self.num_classes, out2, in2))


    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        # x = B,N,C
        # y = B
        if y is not None:
            M_fc1 = self.mask_fc1[y] #B, out, in
            M_fc2 = self.mask_fc2[y] #B, out, in
        else:
            assert 1 == 2, "todo for inference"
            
        W1 = self.fc1.weight.unsqueeze(0) * M_fc1 #B, out1, in1 (unsqeeze to add batch dim to fc1 weights)
        print("fc1 w", self.fc1.weight.shape)
        b1 = self.fc1.bias # out1
        x = torch.einsum('bni,boi->bno', x, W1) + b1.unsqueeze(0).unsqueeze(0) #B, N, out1
        # einsum bc we have weights with batch dim, so need batch mat mul
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        W2 = self.fc2.weight.unsqueeze(0) * M_fc2 #B, out2, in2 (=out1)
        b2 = self.fc2.bias
        x = torch.einsum('bni,boi->bno', x, W2) + b2.unsqueeze(0).unsqueeze(0)
        x = self.drop2(x)
        return x # B, N, C

class MaskedDeiT(nn.Module):
    """
    deit with masked attn
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes

        #make module lists w our wrapped modules
        self.masked_attn = nn.ModuleList()
        self.masked_mlp = nn.ModuleList()
        for blk in self.model.blocks:
            self.masked_attn.append(MaskedAttn(blk.attn, num_classes = self.num_classes))
            self.masked_mlp.append(MaskedMlp(blk.mlp, num_classes = self.num_classes))
            

        # turn off original modules
        for blk in self.model.blocks:
            for p in blk.attn.parameters(): p.requires_grad = False
            for p in blk.mlp.parameters(): p.requires_grad = False


    def forward_features(self, x, y=None):
        #start is normal
        B = x.shape[0] # batch size
        x = self.model.patch_embed(x)
        cls_tok = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1) 
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        #here we use the correct modules from the lists in place of the original modules
        for i, blk in enumerate(self.model.blocks):
            attn_out = self.masked_attn[i](blk.norm1(x), y)
            x = x + blk.drop_path1(blk.ls1(attn_out))
            
            mlp_out = self.masked_mlp[i](blk.norm2(x), y)
            x = x + blk.drop_path2(blk.ls2(mlp_out))
            
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x, y=None):
        x = self.forward_features(x, y)
        return self.model.head(x)
    

