import math
import sys
import time

import cv2
import os
import numpy as np
from tqdm import tqdm
import json


def label_mapping(label_path, save_path, category_dict):
    for index, label in tqdm(enumerate(os.listdir(label_path))):
        ana = cv2.imread(os.path.join(label_path, label), -1)
        ana = cv2.cvtColor(ana, cv2.COLOR_BGR2GRAY)
        gray_label = np.zeros(ana.shape[:2])
        for key, value in category_dict.items():
            gray_label[np.where(ana == value)] = key

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cv2.imwrite(os.path.join(save_path, label), gray_label)


def load_json_file(json_path):
    _category_dict = {}
    with open(json_path, 'r', encoding='utf8') as fp:
        category_dict = json.load(fp)
        for index, item in enumerate(category_dict['labels']):
            color = item['color']
            value = np.array([[color]]).astype(np.uint8)
            value = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)

            if not value in _category_dict.values():
                _category_dict[index] = value[0][0]

    return _category_dict


if __name__ == '__main__':
    json_file = '/data/fpc/projects/adaspace_validation/ProcessingTools/label_helper.json'
    category_dict = load_json_file(json_file)

    split = ['training', 'validation']

    for i in split:
        label_file = '/data/fpc/data/Mapillaryv1_2/{}/labels'.format(i)
        save_file = '/data/fpc/data/Mapillaryv1_2/{}/labels_'.format(i)
        print(label_file)
        label_mapping(label_file, save_file, category_dict)
