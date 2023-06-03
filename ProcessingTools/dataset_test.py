import os
import sys
import cv2
import random
import shutil
from tqdm import tqdm

def check_file(path):
    if not os.path.exists(path):
        os.mkdir(path)

dir = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_256_100w'

split = ['train', 'val']

output_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/test/'
check_file(output_path)
sample_nums = 20480

for item in split:
    if item == 'val':
        sample_nums = sample_nums / 10
    index = 0
    image_path = os.path.join(dir, item, 'images')
    label_path = os.path.join(dir, item, 'labels')

    item_out_file = os.path.join(output_path, item)
    check_file(item_out_file)

    image_out_file = os.path.join(item_out_file, 'images')
    label_out_file = os.path.join(item_out_file, 'labels')
    check_file(image_out_file)
    check_file(label_out_file)

    for name in tqdm(os.listdir(image_path)):
        if index > sample_nums:
            break
        else:
            index += 1

        image_input_path = os.path.join(image_path, name)
        label_input_path = os.path.join(label_path, name)

        image_out_path = os.path.join(image_out_file, name)
        label_out_path = os.path.join(label_out_file, name)

        shutil.copy(image_input_path, image_out_path)
        shutil.copy(label_input_path, label_out_path)