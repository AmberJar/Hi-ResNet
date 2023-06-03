import torch
import os
import cv2
import numpy as np
import time
from collections import Counter
import shutil
def show_labels(label_path,save_path):
    x = [0]*7
    find = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    airplane_labels = 0
    for root, dirs, files in os.walk(label_path):
        start = time.time()
        img_files = [file_name for file_name in files]
        for i in img_files:
            img = cv2.imread(os.path.join(root, i), -1)
            imgs = np.asarray(img, dtype=np.int8).flatten()
            for j in range(7):
                x[j] += Counter(imgs)[j]
            # if find != 0:
            #     print(i)
            #     print(y)
            #     airplane_labels += 1
            #     img_path = os.path.join()
            #     shutil.copy(os.path.join(root, i), save_path)
            print(time.time()-start)
            # if x > 0:
            #      print('here')
    print(x)

    # print(airplane_labels)
            # print(y)
def change_labels(labels_path,save_path):
    keep_list = [1, 2, 3, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(labels_path):
        label_files = [files_name for files_name in files]
        start = time.time()
        for i in label_files:
            img = cv2.imread(os.path.join(root, i), -1)
            for j in range(16):
                if j not in keep_list:
                    img[img == j] = 0
                else:
                    print(keep_list.index(j))
                    if j == 2:
                        img[img == j] = 2
                    elif j == 3:
                        img[img == j] = 2
                    elif j == 5:
                        img[img == j] = 3
                    elif j == 6 or j == 8:
                        img[img == 6 or img == 8] = 4
                    elif j == 7:
                        img[img == j] = 5
                    elif j == 9:
                        img[img == j] = 6
                    elif j == 10 or j == 11 or j == 12:
                        img[img == j] = 7
                    elif j == 13 or j == 14 or j == 15:
                        img[img == j] = 8
                    else:
                        img[img == j] = keep_list.index(j) + 1
            cv2.imwrite(os.path.join(save_path, i), img)
            print(time.time()-start)

def normalmask_filters(labels_path,save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(labels_path):
        start = time.time()
        img_files = [file_name for file_name in files]
        for i in img_files:
            img = cv2.imread(os.path.join(root, i), -1)
            imgs = np.asarray(img, dtype=np.int8).flatten()
            sum = 0
            for j in range(1, 7):
                sum += Counter(imgs)[j]
            if sum != 0:
                cv2.imwrite(os.path.join(save_path, i), img)
            print(time.time() - start)

def copy_images(img_path,label_path,save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    start = time.time()
    for root, dirs, files in os.walk(label_path):
        labels = [img for img in files]
    for root, dirs, files in os.walk(img_path):
        imgs = [x for x in files]
        for i in imgs:
            find = i.split('.')[0] + '.png'
            if find in labels:
                print(find)
                src_path = os.path.join(root, i)
                shutil.copy(src_path, save_path)
            print(time.time() - start)



if __name__ == '__main__':
    # change_labels(r'/data/chenyuxia/GID/labels_', r'/data/chenyuxia/GID/GID_labels')
    show_labels(r'/data/chenyuxia/NAIC/labels_/NAIC/all_labels_', r'/data/chenyuxia/NAIC/labels_/airplane_labels')
    # normalmask_filters(r'/data/chenyuxia/NAIC/labels_/change_labels', r'/data/chenyuxia/NAIC/labels_/labels_filtered')
    # copy_images(r'/data/chenyuxia/NAIC/labels_/images_filtered', r'/data/chenyuxia/NAIC/labels_/airplane_label', r'/data/chenyuxia/NAIC/labels_/airplane_images')

