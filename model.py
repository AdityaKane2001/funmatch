import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import set_seed
set_seed(42)

class ViT(nn.Module):
    def __init__(self, add_linear=0):
        super().__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()
        self.lins = [nn.Identity()]
        if self.add_linear > 0:
            self.lins = [nn.Linear(768, 768) for _ in range(self.add_linear)]

    def forward(self, x):
        x = self.vit(x)
        if self.add_linear > 0:
            for elem in self.lins:
                x = elem(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, add_linear=0):
        super().__init__()
        self.cnvnxt = models.convnext_tiny(pretrained=False)
        self.cnvnxt.classifier[2] = nn.Identity()
        self.lins = [nn.Identity()]
        if self.add_linear > 0:
            self.lins = [nn.Linear(768, 768) for _ in range(self.add_linear)]

    def forward(self, x):
        x = self.cnvnxt(x)
        if self.add_linear > 0:
            for elem in self.lins:
                x = elem(x)
        return x