import os
from base import PotsdamDataset, BaseDataLoader
import numpy as np
import torch
from PIL import Image
from glob import glob
from utils import palette
import cv2
import random
from torch.utils.data.distributed import DistributedSampler
import scipy.io as io


class Potsdam(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, num_classes, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, random_aug=False, return_id=False, segfix=False):
        # self.MEAN = [0.337243, 0.333592, 0.360099]
        # self.STD = [0.131107, 0.133054, 0.128429]

        # self.MEAN = [0.336968, 0.333265, 0.359805]
        # self.STD = [0.128367, 0.130168, 0.125749]
        # self.MEAN = [0.336968, 0.333265, 0.359805]
        # self.STD = [0.118115, 0.120541, 0.116598]
        #vaihingen
        self.MEAN = [0.463633, 0.316652, 0.320528]
        self.STD = [0.203334, 0.135546, 0.140651]
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'num_classes': num_classes,
        }
        self.dataset = PotsdamDataset(**kwargs)
        self.train_sampler = DistributedSampler(self.dataset, shuffle=True)

        super(Potsdam, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle, sampler=self.train_sampler)
