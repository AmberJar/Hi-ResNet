import os
import torch
import numpy as np
import cv2
import shutil
import time
from collections import Counter
def show_labels(labels_path,new_path):
    x = 0
    y = 0
    z = 0
    for root, dirs, files in os.walk(labels_path):
        start = time.time()
        img_files = [file_name for file_name in files]
        for i in img_files:
            img = cv2.imread(os.path.join(root, i), -1)
            # imgs = np.asarray(img, dtype=np.int8).flatten()
            img[img == 16] = 5
            img[img == 17] = 6
            # for j in range(7, 18):
            #     locations = np.all(img == j, axis=-1)
            #     print(locations)
            #     img[locations] = j - 2
            # print(os.path.join(new_path, i.split('.')[0] + '_text.png'))
            cv2.imwrite(os.path.join(new_path, i.split('.')[0] + '_text.png'), img)
            # x += Counter(imgs)[5]
            # y += Counter(imgs)[6]
            # z += Counter(imgs)[16]
            print(time.time() - start)
            # if z > 0 :
            #     print('here')
            # print(x)
            # print(y)

if __name__ == '__main__':
    print(show_labels(r'/data/chenyuxia/NAIC/NAICdata/train/labels', r'/data/chenyuxia/NAIC/text'))

