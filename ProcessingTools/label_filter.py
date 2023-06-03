###############
# 根据面积筛选
##############
import sys

import cv2
from collections import Counter
import os
import numpy as np
from PIL import Image
import time
import shutil
from tqdm import tqdm
import torch


def label_filter(_images_path, _labels_path, _img_save_path, _label_save_path):
    k = 0
    for i in tqdm(os.listdir(_images_path)):
        j = i[:-3] + 'png'
        image_path = os.path.join(_images_path, i)
        label_path = os.path.join(_labels_path, j)

        image_save = os.path.join(_img_save_path, i)
        label_save = os.path.join(_label_save_path, j)

        # image_save_1 = os.path.join('/data/fpc/data/GID_text/train/images_filtered_80', i)
        # label_save_1 = os.path.join('/data/fpc/data/GID_text/train/labels_filtered_80', j)
        #
        # image_save_2 = os.path.join('/data/fpc/data/GID_text/train/images_filtered_70', i)
        # label_save_2 = os.path.join('/data/fpc/data/GID_text/train/labels_filtered_70', j)

        if not os.path.exists(img_save_path):  # 如果路径不存在
            os.makedirs(img_save_path)
        if not os.path.exists(label_save_path):  # 如果路径不存在
            os.makedirs(label_save_path)

        # if not os.path.exists(image_save_1):  # 如果路径不存在
        #     os.makedirs(image_save_1)
        # if not os.path.exists(label_save_1):  # 如果路径不存在
        #     os.makedirs(label_save_1)
        #
        # if not os.path.exists(image_save_2):  # 如果路径不存在
        #     os.makedirs(image_save_2)
        # if not os.path.exists(label_save_2):  # 如果路径不存在
        #     os.makedirs(label_save_2)

        ana = cv2.imread(label_path, 0)
        if not 65 in ana:
            continue
        pixels = ana.shape[0] * ana.shape[1]
        ana = torch.tensor(ana).flatten().to(torch.cuda.current_device())
        categories_num_list = torch.bincount(ana)

        # ana[ana == 255] = 1
        # ratio = ana.sum() / (ana.shape[0] * ana.shape[1])

        ratio_0 = categories_num_list[-1] / pixels

        if ratio_0 < 0.90:
            shutil.copy(image_path, image_save)
            shutil.copy(label_path, label_save)
            k += 1

        # if ratio < 0.80:
        #     shutil.copy(image_path, image_save_1)
        #     shutil.copy(label_path, label_save_1)
        #
        # if ratio < 0.70:
        #     shutil.copy(image_path, image_save_2)
        #     shutil.copy(label_path, label_save_2)
    print(k)


def ratio_filter(_images_path, _labels_path, _img_save_path, _label_save_path):
    if not os.path.exists(_img_save_path):
        os.mkdir(_img_save_path)
    if not os.path.exists(_label_save_path):
        os.mkdir(_label_save_path)

    for i in tqdm(os.listdir(_images_path)):
        number = np.random.randint(5)

        if number < 3:
            shutil.copy(os.path.join(_images_path, i), os.path.join(_img_save_path, i))
            shutil.copy(os.path.join(_labels_path, i), os.path.join(_label_save_path, i))


if __name__ == "__main__":
    images_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/train/images_448'
    labels_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/train/labels_448'
    img_save_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset/train/images'
    label_save_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset/train/labels'
    split = ['train', 'val']

    # label_filter(images_path, labels_path, img_save_path, label_save_path)

    for i in split:
        images_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/{}/images_448'.format(i)
        labels_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/{}/labels_448'.format(i)
        img_save_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/3o5_trainset/images'
        label_save_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/3o5_trainset/labels'
        ratio_filter(images_path, labels_path, img_save_path, label_save_path)
