import time

import torch
import os
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
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for filename in os.listdir(file_path):
        names = filename.split('_')
        new_name = names[0] + '_sat.png'
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
        img = cv2.imread(os.path.join(pic_path, i))
        print(img.shape)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # mask的采样
        j = i[:-3] + 'png'
        label = cv2.imread(os.path.join(label_path, j))
        label = cv2.resize(label, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        label[label < 127] = 0
        label[label >= 127] = 255

        cv2.imwrite(os.path.join(pic_save, i), img)
        cv2.imwrite(os.path.join(label_save, j), label)

        print(time.time() - start)


def images_cut(img_path, mask_path, image_save_path, label_save_path):
    for i in os.listdir(img_path):
        pic = cv2.imread(os.path.join(img_path, i))
        mask = cv2.imread(os.path.join(mask_path, i))

        pic = pic[56:456, 56:456]
        mask = mask[56:456, 56:456]
        print(pic.shape)
        cv2.imwrite(os.path.join(image_save_path, i), pic)
        cv2.imwrite(os.path.join(label_save_path, i), mask)


if __name__ == "__main__":
    file_path = '/data/fpc/data/deep_512/images_512/'
    save_path = '/data/fpc/data/deep_512/labels_512/'

    # file_rename(file_path, save_path)
    # pic_path = '/data/fpc/data/archive/images'
    # label_path = '/data/fpc/data/archive/labels'
    #
    # pic_save = '/data/fpc/data/archive/images_resize'
    # label_save = '/data/fpc/data/archive/labels_resize'
    #
    # if not os.path.exists(pic_save):
    #     os.makedirs(pic_save)
    # if not os.path.exists(label_save):
    #     os.makedirs(label_save)
    #
    # images_resize(pic_path, label_path, pic_save, label_save)

    img_path = '/data/fpc/data/deep_512/images_512'
    mask_path = '/data/fpc/data/deep_512/labels_512'
    image_save_path = '/data/fpc/data/deep_512/road_400/images'
    label_save_path = '/data/fpc/data/deep_512/road_400/labels'
    images_cut(img_path, mask_path, image_save_path, label_save_path)