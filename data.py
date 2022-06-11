from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import set_seed

set_seed(42)

class MixupDataset(Dataset):
    def __init__(self, ds, mixup_rate=0.2, mode="train", num_classes=10):
        self.ds = ds
        self.mixup_rate = mixup_rate
        self.mode = mode
        self.num_classes = num_classes
        
        self.totensor = T.ToTensor()

        self.train_transforms = T.Compose([   
            T.RandomResizedCrop(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.eval_transforms = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        lbl = F.one_hot(lbl, num_classes=self.num_classes)
        img = self.totensor(img)

        if self.mode == "train" and idx % 3 == 0:
            random_idx = np.random.randint(0, len(self.ds))
            img1, lbl1 = self.ds[random_idx]
            img1 = self.totensor(img1)
            lbl1 = F.one_hot(lbl1, num_classes=10)

            img = img * (1 - self.mixup_rate) + img1 * self.mixup_rate
            lbl = lbl * (1 - self.mixup_rate) + lbl1 * self.mixup_rate

            return self.train_transforms(img), lbl
        else:
            return self.eval_transforms(img), lbl

def get_dataloaders(mixup_rate=0.2, batch_size=128):   
    train_ds = CIFAR10("cifar10", train=True, download=True)
    train_ds = MixupDataset(train_ds, mode="train")

    test_ds = CIFAR10("cifar10", train=False, download=True, mixup_rate=mixup_rate)
    val_ds, test_ds = train_set, val_set = torch.utils.data.random_split(test_ds, [int(len(test_ds) * 0.5), len(test_ds) - int(len(test_ds) * 0.5)]) 
    val_ds = MixupDataset(val_ds, mode="eval", mixup_rate=mixup_rate)
    test_ds = MixupDataset(test_ds, mode="eval", mixup_rate=mixup_rate)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False), DataLoader(test_ds, batch_size=batch_size, shuffle=False)
