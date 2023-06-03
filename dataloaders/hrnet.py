import os
from base import BaseDataSet, BaseDataLoader
import numpy as np
import torch
from PIL import Image
from glob import glob
from utils import palette
import cv2
import random
from torch.utils.data.distributed import DistributedSampler
import scipy.io as io


class HRNetDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 7
        self.palette = palette.AdaSpaceMaps_palette

        # water
        # self.huawei_id_to_trainId = {2: 1, 3: 2, 4: 3, 5: 4}

        super(HRNetDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, self.split, "images")
        self.label_dir = os.path.join(self.root, self.split, "labels")
        # print(self.image_dir, "\n", self.label_dir)
        self.files = [os.path.basename(path).split('.')[0] for path in
                      glob(self.image_dir + '/*.png')]
        # random.shuffle(self.files)
        # print("self.files", self.files)

    def _load_data(self, index):
        image_id = self.files[index]
        label_id = image_id
        offset_id = image_id
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, label_id + '.png')

        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.asarray(cv2.imread(label_path, 0), dtype=np.int32)

        if len(label.shape) == 3:
            label = label[:, :, 0]

        # for k, v in self.huawei_id_to_trainId.items():
        #    label[label == k] = v

        return image, label, image_id


class HRNet(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, num_workers=1, shuffle=False, val_split=None, random_aug=False):
        self.MEAN = [0.294068, 0.302425, 0.316089]
        self.STD = [0.120058, 0.101178, 0.107847]

        # pretrain Mapillary 256 100w
        # [0.428035, 0.482522, 0.469209]
        # [0.110374, 0.108241, 0.109868]

        # pretrain Maplillary 376
        # [0.432247, 0.48861, 0.474542]
        # [0.125939, 0.125201, 0.125824]

        # GID3W + NAIC4W
        # [0.409974, 0.433444, 0.446292]
        # [0.187452, 0.144408, 0.161365]

        # GID 16 aug
        # self.MEAN = [0.342171, 0.332106, 0.366867]
        # self.STD = [0.185678, 0.1657, 0.181847]

        # GID 16 only train aug
        # [0.337986, 0.328924, 0.361587]
        # [0.178694, 0.15631, 0.173528]

        # LoveDA train aug 6w
        # [0.294068, 0.302425, 0.316089]
        # [0.120058, 0.101178, 0.107847]

        # LoveDA train aug 12w
        # [0.293982, 0.299323, 0.314536]
        # [0.113959, 0.097802, 0.103131]

        # loveDA train aug 6w-sample
        # [0.293782, 0.299317, 0.314574]
        # [0.113962, 0.097799, 0.103167]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'random_aug': random_aug,
        }
        self.dataset = HRNetDataset(**kwargs)
        self.train_sampler = DistributedSampler(self.dataset, shuffle=True)

        super(HRNet, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle, sampler=self.train_sampler)
