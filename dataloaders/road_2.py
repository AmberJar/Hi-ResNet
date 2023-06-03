import os
from base import BaseDataSet, BaseDataLoader
import numpy as np
import torch
from PIL import Image
from glob import glob
from utils import palette
import cv2
from torch.utils.data.distributed import DistributedSampler


class Road2Dataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.AdaSpaceMaps_palette

        # water
        # self.huawei_id_to_trainId = {2: 1, 3: 2, 4: 3, 5: 4}

        super(Road2Dataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, self.split, "images")
        self.label_dir = os.path.join(self.root, self.split, "labels")
        # print(self.image_dir, "\n", self.label_dir)
        self.files = [os.path.basename(path).split('.')[0] for path in
                      glob(self.image_dir + '/*.png')]
        # print("self.files", self.files)

    def _load_data(self, index):
        image_id = self.files[index]
        label_id = image_id
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, label_id + '.png')
        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.asarray(cv2.imread(label_path, -1), dtype=np.int32)
        label[label == 255] = 1

        # water
        # label[label <= 5.5] = 0
        # label[label > 5.5] = 1
        # label[label == 5] = 1
        # print(label_id, label.shape)
        if len(label.shape) == 3:
            label = label[:, :, 0]

        # for k, v in self.huawei_id_to_trainId.items():
        #    label[label == k] = v

        return image, label, image_id


class Road2(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):
        self.MEAN = [0.442722, 0.378454, 0.43694]
        self.STD = [0.183933, 0.163688, 0.16578]

        # first road outs
        # [0.402568, 0.406539, 0.401525]
        # [0.135746, 0.140629, 0.127753]
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        self.dataset = Road2Dataset(**kwargs)
        # self.train_sampler = DistributedSampler(self.dataset)

        super(Road2, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle)
