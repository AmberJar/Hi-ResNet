import time

import torch
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision import utils as vutils

'''
如果要缩小图像，建议选择：cv2.INTER_AREA；
如果要放大图像，cv2.INTER_CUBIC效果更好但是速度慢，
cv2.INTER_LINEAR效果尚可且速度快。
进行缩放时， dsize和fx、fy 二选一即可。
'''


def file_rename(file_path, save_path):
    for filename in os.listdir(file_path):
        names = filename.split('_')
        new_name = names[0] + 'sat.png'
        # print(new_name)
        # img = cv2.imread(os.path.join(filename, file_path))
        # print(img.shape)
        src = os.path.join(file_path, filename)
        dst = os.path.join(save_path, new_name)
        os.rename(src, dst)


def images_resize(pic_path, label_path, pic_save, label_save):
    path_list = os.listdir(pic_path)
    start = time.time()
    for i in path_list:
        # 图像的采样
        img = cv2.imread(os.path.join(pic_path, i), -1)
        print(img.shape)
        img = cv2.resize(img, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        # mask的采样
        j = i[:-3] + 'png'
        label = cv2.imread(os.path.join(label_path, j), 0)
        label = cv2.resize(label, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        label = np.rint(label)
        label[label > 16] = 16

        # label[label < 127] = 0
        # label[label >= 127] = 255

        cv2.imwrite(os.path.join(pic_save, i), img.astype(np.uint8))
        cv2.imwrite(os.path.join(label_save, j), label.astype(np.uint8))

        print(time.time() - start)


if __name__ == "__main__":
    # file_path = '/data/fpc/data/archive/labels/'
    # save_path = '/data/fpc/data/archive/labels_/'
    # file_rename(file_path, save_path)
    pic_path = '/data/chenyuxia/17classes_data/all_images'
    label_path = '/data/chenyuxia/17classes_data/labels_text'

    pic_save = '/data/chenyuxia/17classes_data/images_resize'
    label_save = '/data/chenyuxia/17classes_data/labels_resize'

    if not os.path.exists(pic_save):
        os.makedirs(pic_save)
    if not os.path.exists(label_save):
        os.makedirs(label_save)

    images_resize(pic_path, label_path, pic_save, label_save)