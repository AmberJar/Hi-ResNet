import shutil

import cv2
import os
import random
from tqdm import tqdm

img_file = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376/val/images'
label_file = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376/val/labels'
img_save_file = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376/val/images_'
label_save_file = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376/val/labels_'

if not os.path.exists(img_save_file):
    os.mkdir(img_save_file)
if not os.path.exists(label_save_file):
    os.mkdir(label_save_file)
k = 0
for i in tqdm(os.listdir(img_file)):
    if k % 2 == 0:
        shutil.copy(os.path.join(img_file, i), os.path.join(img_save_file, i))
        shutil.copy(os.path.join(label_file, i), os.path.join(label_save_file, i))
    k += 1

