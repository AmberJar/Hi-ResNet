from itertools import dropwhile
import os
import sys

import numpy
from skimage.io import imsave
from torch.utils.tensorboard import SummaryWriter

sys.path.append("./")
from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

HUAWEI = "huawei"
GOOGLE = "test-google"
CHN = "CHN6-CUG"
TIANCHI = "tianchi"
CROWED_AI = "crowed_ai"

DATASET = {HUAWEI, GOOGLE, CHN, TIANCHI, CROWED_AI}
AIM_COLOR = {"bg": [0, 0, 0],
             "road": [255, 255, 255],
             "water": [0, 0, 255],
             "soil": [128, 64, 0],
             "building": [255, 0, 0],
             "plant": [0, 255, 0]}


# AIM_COLOR_8bits = {"bg": 0,
#                    "road": 1,
#                    "water": 2,
#                    "building": 3,
#                    "soil": 4,
#                    "plant": 5}


class AdaspaceDataset(BaseDataSet):
    """
    AdaspaceMps dataset
    @author: xu
    """

    def __init__(self, **kwargs):
        self.VOC_COLORMAP = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [128, 64, 0], [255, 0, 0], [0, 255, 0]]

        self.VOC_CLASSES = ['background', 'road', 'waterbody', 'soil', 'building', 'plants']
        # self.colormap = colormap
        self.id_to_trainId = {4: 0}  # 改后者为2 建筑， 0 背景 # 1:0在前如果是分类building。
        # self.huawei_id_to_trainId = {2:0, 3:0, 4:0, 5:0}
        self.TIANCHI_id_to_trainId = {255: 1}  # 天池数据集中只有建筑和背景，且原建筑为255.
        self.image_dir_list = []
        self.label_dir_list = []
        self.num_classes = 3
        self.palette = palette.AdaSpaceMaps_palette
        super(AdaspaceDataset, self).__init__(**kwargs)

    def _set_files(self):
        data_name = self.set_name
        if data_name == 'uinfy_datasets':
            if self.split in ["train"]:
                self.image_dir = os.path.join(self.root, self.split, 'images')
                # images path ./datasets/train/images or ./datasets/val/imgages
                self.label_dir = os.path.join(self.root, self.split, 'gt')
                # label path ./datasets/train/gt or ./datasets/val/gt
                self.files = [os.path.basename(path).split('.')[0] for path in
                              glob(self.image_dir + '/*.jpg')]
                # contain total train or val images stem.
            else:
                raise ValueError(f"Invalid split name {self.split}")
        if data_name == CHN:
            if self.split in ["train", "val"]:
                self.image_dir = os.path.join(self.root, self.split, 'images')
                # images path ./datasets/train/images or ./datasets/val/imgages
                self.label_dir = os.path.join(self.root, self.split, 'gt')
                # label path ./datasets/train/gt or ./datasets/val/gt
                self.files = [os.path.basename(path).split('.')[0] for path in
                              glob(self.image_dir + '/*.jpg')]
                # contain total train or val images stem.
            else:
                raise ValueError(f"Invalid split name {self.split}")
        elif data_name == GOOGLE:
            # self.id_to_trainId = {4:0} # 改后者为2 建筑， 0 背景
            # self.image_dir_list = []
            # self.label_dir_list = []
            if self.split == 'train':
                for self.location in ["berlin", "chicago", "paris", "zurich"]:  # 若没有paris要加上paris。

                    self.image_dir = os.path.join(self.root, self.location, "after_cut")
                    self.label_dir = os.path.join(self.root, self.location, "after_cut_labels")

                    length_of_dir = [os.path.basename(path).split('.')[0] for path in
                                     glob(self.image_dir + '/*.png')]

                    self.files += [os.path.basename(path).split('.')[0] for path in
                                   glob(self.image_dir + '/*.png')]
                    for i in range(0, len(length_of_dir)):
                        self.image_dir_list.append(self.image_dir)
                        self.label_dir_list.append(self.label_dir)
            elif self.split == 'val':
                for self.location in ["potsdam", "tokyo"]:

                    self.image_dir = os.path.join(self.root, self.location, "after_cut")
                    self.label_dir = os.path.join(self.root, self.location, "after_cut_labels")

                    length_of_dir = [os.path.basename(path).split('.')[0] for path in
                                     glob(self.image_dir + '/*.png')]

                    self.files += [os.path.basename(path).split('.')[0] for path in
                                   glob(self.image_dir + '/*.png')]
                    for i in range(0, len(length_of_dir)):
                        self.image_dir_list.append(self.image_dir)
                        self.label_dir_list.append(self.label_dir)


            else:
                raise ValueError(f"Invalid split name {self.split}")

        elif data_name == HUAWEI:
            if self.split in ["train", "val"]:
                print(self.root)
                self.image_dir = os.path.join(self.root, self.split, 'images')
                self.label_dir = os.path.join(self.root, self.split, 'labels')
                self.files = [os.path.basename(path).split('.')[0] for path in
                              glob(self.image_dir + '/*.png')]
            else:
                raise ValueError(f"Invalid split name {self.split}")

        elif data_name == TIANCHI:
            if self.split in ["train", "val"]:
                print(self.root)
                self.image_dir = os.path.join(self.root, self.split, 'images')
                self.label_dir = os.path.join(self.root, self.split, 'labels')
                self.files = [os.path.basename(path).split('.')[0] for path in
                              glob(self.image_dir + '/*.png')]
            else:
                raise ValueError(f"Invalid split name {self.split}")

        elif data_name == CROWED_AI:
            if self.split in ["train", "val"]:
                print(self.root)
                self.image_dir = os.path.join(self.root, self.split, 'images')
                self.label_dir = os.path.join(self.root, self.split, 'images')
                self.files = [os.path.basename(path).split('.')[0] for path in
                              glob(self.image_dir + '/*.png')]
            else:
                raise ValueError(f"Invalid split name {self.split}")

        else:
            raise ValueError(f"Invalid data_set name {data_name}")

    def _load_data(self, index):

        data_name = self.set_name

        image_id = self.files[index]

        if data_name == CHN:
            label_id = image_id.replace("sat", "mask")
            image_path = os.path.join(self.image_dir, image_id + '.jpg')
            # train or val images path, example ./datesets/train/xxxxx.jpg
            label_path = os.path.join(self.label_dir, label_id + '.png')
            # train or val gt path, example ./datasets/train/xxxxx.png
            image = np.asarray(Image.open(image_path).convert('RGB'),
                               dtype=np.float32)
            # images for train or val dtype is float32
            label = np.asarray(Image.open(label_path), dtype=np.int32)


        elif data_name == GOOGLE:
            image_dir_id = self.image_dir_list[index]

            label_dir_id = self.label_dir_list[index]
            label_id = image_id.replace("image", "labels")
            image_path = os.path.join(image_dir_id, image_id + '.png')

            label_path = os.path.join(label_dir_id, label_id + '.png')

            image = np.asarray(Image.open(image_path).convert('RGB'),
                               dtype=np.float32)
            # label = cv2.imread(label_path, 0)
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            for k, v in self.id_to_trainId.items():
                label[label == k] = v
            # print(f"label shape {label.shape}")
            # print(f"lable value {label[0:200, 0:200]}")
            # img = Image.open(image_path)
            # img.show()
            # plt.imshow(label)  # huawei数据图像 像素点 300-背景 600-草垛 500-草坪
            # plt.show()

            # label.resize((3000, 3000))
            # image.resize((3000, 3000, 3))

        elif data_name == HUAWEI:
            # image_dir_id = self.image_dir_list[index]

            # label_dir_id = self.label_dir_list[index]
            # label_id = image_id.replace("image", "labels")
            image_path = os.path.join(self.image_dir, image_id + '.png')

            label_path = os.path.join(self.label_dir, image_id + '.png')

            image = np.asarray(Image.open(image_path).convert('RGB'),
                               dtype=np.float32)
            # label = cv2.imread(label_path, 0)
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            label[label == 255] = 1
        elif data_name == TIANCHI:
            # image_dir_id = self.image_dir_list[index]

            # label_dir_id = self.label_dir_list[index]
            # label_id = image_id.replace("image", "labels")
            image_path = os.path.join(self.image_dir, image_id + '.png')

            label_path = os.path.join(self.label_dir, image_id + '.png')

            image = np.asarray(Image.open(image_path).convert('RGB'),
                               dtype=np.float32)
            # label = cv2.imread(label_path, 0)
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            for k, v in self.TIANCHI_id_to_trainId.items():
                label[label == k] = v
            # print(label)
        elif data_name == CROWED_AI:
            # image_dir_id = self.image_dir_list[index]

            # label_dir_id = self.label_dir_list[index]
            # label_id = image_id.replace("image", "labels")
            image_path = os.path.join(self.image_dir, image_id + '.png')

            label_path = os.path.join(self.label_dir, image_id + '.png')

            # label = cv2.imread(label_path, 0)
            label = np.asarray(Image.open(label_path), dtype=np.float32)
            # crowed_ai训练集
            # label[label != 0] = 1

            # 自编码器
            # image = patches(Image.open(image_path), 75)
            # image = np.asarray(image, dtype=np.float32) / 255
            # label = label / 255
        else:
            print("The data name is incorrect.")
            return -1

        return image, label, image_id

    def normal_color(self):
        pass

    # def dataset_judge(self):
    #     set_name = self.set_name
    #     return set_name


class Adaspace(BaseDataLoader):
    def __init__(self, data_dir, set_name, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'set_name': set_name,
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
            'val': val,

        }
        self.set_name = set_name
        self.dataset_judge()
        self.dataset = AdaspaceDataset(**kwargs)
        super(Adaspace, self).__init__(self.dataset, batch_size, shuffle,
                                       num_workers, val_split)

    def dataset_judge(self):
        # set_name = data_dir.split("/")[-1]
        set_name = self.set_name
        if set_name not in DATASET:
            raise ValueError(f"Invalid dataset name")
        return set_name


if __name__ == "__main__":
    # writer = SummaryWriter("dataloader1111")
    data_dir = r"/data/zhangruozhao/tianchi/AerialImageDataset/"
    # # image = np.asarray(Image.open(data_dir).convert('RGB'),
    # # dtype=np.float32)
    # # image1 = cv2.imread(data_dir, 0)  # 以灰度图像读取
    # # print(image1.shape)
    # # image1 = plt.imshow(data_dir, cv2.CV_16UC1)
    # # plt.imshow(image1)  # huawei数据图像 像素点 300-背景 600-草垛 500-草坪
    # # plt.show()
    batch_size = 10
    split = r"val"
    data = Adaspace(data_dir, set_name='tianchi', batch_size=batch_size, split=split)
    # n = 0
    for img, lab in data:
        print(f"{img.shape} {img.dtype}")
        print(f"{lab.shape} {lab.dtype}")
        # lab = lab.unsqueeze(1)
        # print(lab.shape)
        # writer.add_images("demo_image", img, n)
        # writer.add_images("demo_label", lab, n)
        # n += 1
        # print(lab)
        print("----------------------------------")
        # step += 1
