###############
# 根据面积筛选
##############
import cv2
from collections import Counter
import os
import numpy as np
from PIL import Image
import time
import shutil
import random
import albumentations as albu
import os.path as osp
from tqdm import tqdm

def load_img_and_mask(name):
    name_ = name.split('.')[0] + '.png'
    img_name = os.path.join(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/ori/train/images_1024', name)
    mask_name = os.path.join(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/ori/train/masks_1024', name_)
    img = Image.open(img_name).convert('RGB')
    mask = Image.open(mask_name).convert('L')

    return img, mask


def load_mosaic_img_and_mask(img_path):
    img_file = [files for files in os.listdir(img_path)]
    indexes = [random.randint(0, len(img_file) - 1) for _ in range(4)]
    img_a, mask_a = load_img_and_mask(img_file[indexes[0]])
    img_b, mask_b = load_img_and_mask(img_file[indexes[1]])
    img_c, mask_c = load_img_and_mask(img_file[indexes[2]])
    img_d, mask_d = load_img_and_mask(img_file[indexes[3]])
    w = img_a.size[1]
    print(w)
    h = img_a.size[0]

    img_a, mask_a = np.array(img_a), np.array(mask_a)
    img_b, mask_b = np.array(img_b), np.array(mask_b)
    img_c, mask_c = np.array(img_c), np.array(mask_c)
    img_d, mask_d = np.array(img_d), np.array(mask_d)


    start_x = w // 4
    strat_y = h // 4
    # The coordinates of the splice center
    offset_x = random.randint(start_x, (w - start_x))
    offset_y = random.randint(strat_y, (h - strat_y))

    crop_size_a = (offset_x, offset_y)
    crop_size_b = (w - offset_x, offset_y)
    crop_size_c = (offset_x, h - offset_y)
    crop_size_d = (w - offset_x, h - offset_y)

    random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
    random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
    random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
    random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

    croped_a = random_crop_a(image=img_a, mask=mask_a)
    croped_b = random_crop_b(image=img_b, mask=mask_b)
    croped_c = random_crop_c(image=img_c, mask=mask_c)
    croped_d = random_crop_d(image=img_d, mask=mask_d)

    img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
    img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
    img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
    img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

    top = np.concatenate((img_crop_a, img_crop_b), axis=1)
    bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
    img = np.concatenate((top, bottom), axis=0)

    top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
    bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
    mask = np.concatenate((top_mask, bottom_mask), axis=0)
    mask = np.ascontiguousarray(mask)
    img = np.ascontiguousarray(img)

    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    return img, mask


if __name__ == '__main__':
    for i in tqdm(range(2000)):
        img, mask = load_mosaic_img_and_mask(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/ori/train/images_1024')
        # print(os.path.join(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/potsdam/train/augment_img', str(i) + '.tif'))
        img.save(os.path.join(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/potsdam/train/augment_images', str(i) + '.tif'))
        mask.save(os.path.join(r'/mnt/data/chenyuxia/EXPERIMENTAL/potsdam/potsdam/train/augment_masks', str(i) + '.png'))
