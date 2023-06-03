import os
import cv2
import numpy as np
import torch
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth

        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)

        return CE_loss

mask = torch.randn(1, 20, 10)
mask = torch.tensor(mask, dtype=torch.long)
b = torch.randn(1, 3, 20, 10)
b = torch.tensor(b, dtype=torch.float)
ce = CE_DiceLoss()

res = ce(b, mask)

