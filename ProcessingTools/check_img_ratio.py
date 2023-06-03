import os
import sys
from collections import Counter
import numpy as np
from tqdm import tqdm
import cv2


def check_mask_class_ratio(mask_path, num_classes):
    class_dict = {}
    pixel_dict = {}

    for i in range(num_classes):
        class_dict[i] = 0
        pixel_dict[i] = 0

    for j in tqdm(os.listdir(mask_path)):
        input_mask = cv2.imread(os.path.join(mask_path, j))

        class_distribution = Counter(input_mask.flatten())

        for key, value in class_distribution.items():
            class_dict[key] += 1
            pixel_dict[key] += value

    print(class_dict)
    print(pixel_dict)


if __name__ == '__main__':
    msk_path = '/data/fpc/data/GID_text/train/labels'
    classes = 16

    check_mask_class_ratio(msk_path, classes)
