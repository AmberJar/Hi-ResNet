import os
import cv2
import random
import shutil
from tqdm import tqdm

total = 117846
sample_nums = 60000
image_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_256_100w/val/images'
label_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_256_100w/val/labels'
output_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_256_40w/val'

image_out_file = os.path.join(output_path, 'images')
label_out_file = os.path.join(output_path, 'labels')

def check_file(path):
    if not os.path.exists(path):
        os.mkdir(path)

check_file(image_out_file)
check_file(label_out_file)

for index, name in tqdm(enumerate(os.listdir(image_path))):
    if random.random() > 0.4:
        continue
    image_input_path = os.path.join(image_path, name)
    label_input_path = os.path.join(label_path, name)

    image_out_path = os.path.join(image_out_file, name)
    label_out_path = os.path.join(label_out_file, name)

    shutil.copy(image_input_path, image_out_path)
    shutil.copy(label_input_path, label_out_path)


