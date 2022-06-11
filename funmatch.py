import torch
from torch import nn
from torch.nn import functional as F
import wandb

from data import get_dataloaders
from utils import set_seed
from model import ConvNeXt, ViT

teacher = ViT()
student = ConvNeXt()

loss_fn = nn.KLDivLoss()

NUM_EPOCHS = 1000
