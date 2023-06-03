import sys
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import os

# labels_deepglobe_road = {"road": [255, 255, 255], "backgroud":[0, 0, 0]}

labels_mapping = {"background": 0, "agriculture": 1, "water": 2, "forest": 3, "barred": 4}

labels_HUAWEI_1 = {"background": [2], "water": [4], "forest": [3], "barred": [5], "agriculture": [1]}
labels_HUAWEI_2 = {"background": [0]}
labels_HUAWEI_3 = {"background": [6]}
# labels_HUAWEI_4 = {"water": 12}
# labels_HUAWEI_5 = {"forest": 13}
# labels_HUAWEI_6 = {"barred": 14}
# labels_HUAWEI_7 = {"backgroud": [7]}


# labels = [labels_HUAWEI_1]
labels = [labels_HUAWEI_1, labels_HUAWEI_2, labels_HUAWEI_3]


def convert_label(label_path, labels, labels_mapping):
    src_label = np.asarray(Image.open(label_path), dtype=np.int32)
    src_label = src_label.reshape((*src_label.shape[0:2], -1))
    dst_label = np.zeros(src_label.shape[0:2], dtype=np.uint8)
    for i in labels:
        for label_k, label_v in i.items():
            masks = []
            for channel_i, pixel_value in enumerate(label_v):
                masks.append(src_label[..., channel_i] == pixel_value)
            mask = masks[0]
            if len(masks) > 1:
                for mask_i in masks[1:]:
                    mask *= mask_i
                #             print(label_k)
            dst_label[mask] = labels_mapping.get(label_k, labels_mapping.get("backgroud"))

    return dst_label


def save_labels(path, image, name):
    cv2.imwrite(os.path.join(path, name + ".png"), image)


def pre_color(path, save_path):
    #     print(path)
    back_times = 0
    for i in os.listdir(path):
        img = convert_label(os.path.join(path, i), labels, labels_mapping)
        back_times += 1
        save_labels(save_path, img, i[0:-4])

    print(back_times)


def change_labels(path, save_path):
    start = time.time()
    for i in os.listdir(path):
        label_path = os.path.join(path, i)
        ana = cv2.imread(label_path)

        ana[ana == 255] = 1

        cv2.imwrite(os.path.join(save_path, i), ana)

    print(time.time() - start)


def change_labels_(labels_path, labels_save_path):
    if not os.path.exists(labels_save_path):
        os.mkdir(labels_save_path)

    for i in tqdm(os.listdir(labels_path)):
        label = cv2.imread(os.path.join(labels_path, i), 0)
        label[label == 0] = -1
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 2
        label[label == 4] = 3
        label[label == 5] = 4
        label[label == 6] = 5
        label[label == 7] = 6

        cv2.imwrite(os.path.join(labels_save_path, i), label)



if __name__ == "__main__":
    #     print("hh")
    label = "/data/fpc/data/love_DA/loveDA/val/labels"
    ana_save_path = "/data/fpc/data/love_DA/loveDA/val/labels_"

    change_labels_(label, ana_save_path)