import os
from base import LovedaDataset, BaseDataLoader
import numpy as np
import torch
from PIL import Image
from glob import glob
from utils import palette
import cv2
import random
from torch.utils.data.distributed import DistributedSampler
import scipy.io as io


class Loveda(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, num_classes, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, random_aug=False, return_id=False, segfix=False):

        #loveda_no_augment
        # self.MEAN = [0.280082, 0.299398, 0.307035]
        # self.STD = [0.127366, 0.109451, 0.115518]

        # 7w NAIC
        # self.MEAN = [0.29446, 0.300793, 0.315524]
        # self.STD = [0.115252, 0.095638, 0.1026]

        # road
        self.MEAN = [0.442722, 0.378454, 0.43694]
        self.STD = [0.183933, 0.163688, 0.16578]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'num_classes': num_classes,
        }
        self.dataset = LovedaDataset(**kwargs)
        self.train_sampler = DistributedSampler(self.dataset, shuffle=True)

        super(Loveda, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle, sampler=self.train_sampler)
