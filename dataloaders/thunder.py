import os
from base import BaseDataSetVal, BaseDataLoaderVal
import numpy as np
import torch
from PIL import Image
from glob import glob
from utils import palette
import cv2
from skimage import morphology
import random


class ThunderDataset(BaseDataSetVal):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.AdaSpaceMaps_palette

        # water
        # self.huawei_id_to_trainId = {2: 1, 3: 2, 4: 3, 5: 4}

        super(ThunderDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, self.split, "images")
        self.label_dir = os.path.join(self.root, self.split, "labels")

        # print(self.image_dir, "\n", self.label_dir)
        self.files = [os.path.basename(path).split('.')[0] for path in
                      glob(self.image_dir + '/*.png')]
        # print("self.files", self.files)

    def extract_skeleton(self, gray):
        # img = cv2.imread(gray, cv2.IMREAD_GRAYSCALE)
        # output_path = r'E:\AllProjects\gis_process\cache_output\tmp_res\tmp2.png'

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

        # img = cv2.erode(img, kernel, iterations=1)
        #
        # dilation = cv2.dilate(img, kernel, iterations=1)

        # 開運算
        img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        skeleton = morphology.skeletonize(img, method='zhang')

        return skeleton.astype(np.uint8)

    def _load_data(self, index):
        image_id = self.files[index]
        label_id = image_id
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, label_id + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = cv2.imread(label_path, 0)
        label[label == 255] = 1

        if len(label.shape) == 3:
            label = label[:, :, 0]

        mask = label.copy()
        skeleton = label.copy()
        skeleton = self.extract_skeleton(skeleton)
        road_points = np.where(skeleton == 1)
        # print("road_points", road_points, np.sum(mask))
        point_nums = len(road_points[0])
        # percent = point_nums / (skeleton.shape[0] * skeleton.shape[1])
        size = np.random.randint(7, 13)
        break_switch = False
        k = 0
        while (True):
            rand_list = np.random.randint(0, high=point_nums, size=size, dtype='int')
            x = [(road_points[0][i], road_points[1][i]) for i in rand_list]

            q = 0
            for j in x:
                if 30 < j[0] < skeleton.shape[0] - 1 and 30 < j[1] < skeleton.shape[1] - 1:
                    # print(skeleton_[j[0], j[1]])
                    q += 1

            if q == size or k > 1000:
                break
            else:
                k += 1

        # 接下來擴展生成正方形
        sample_list = [25, 30, 35, 40, 45]
        for ele in x:
            side = random.sample(sample_list, 1)
            side = side[0]
            x1 = ele[0] - side
            if x1 < 0: x1 = 0
            if x1 > skeleton.shape[1]: x1 = skeleton.shape[1] - 2

            x2 = ele[0] + side
            if x2 < 0: x2 = 0
            if x2 > skeleton.shape[1]: x2 = skeleton.shape[1] - 2

            y1 = ele[1] - side
            if y1 < 0: y1 = 0
            if y1 > skeleton.shape[0]: y1 = skeleton.shape[0] - 2

            y2 = ele[1] + side
            if y2 < 0: y2 = 0
            if y2 > skeleton.shape[0]: y2 = skeleton.shape[0] - 2

            mask[x1:x2, y1:y2] = 0

        mask = np.expand_dims(mask, axis=-1)
        pic = np.concatenate([image, mask*255], axis=-1)

        # for k, v in self.huawei_id_to_trainId.items():
        #    label[label == k] = v

        return pic, label, image_id


class Thunder(BaseDataLoaderVal):
    def __init__(self, data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):
        self.MEAN = [0.435513, 0.486176, 0.488189]
        self.STD = [0.27083, 0.238017, 0.231599]

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
        self.dataset = ThunderDataset(**kwargs)
        super(Thunder, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle)
