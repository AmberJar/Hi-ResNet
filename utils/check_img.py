import os
from PIL import Image
import time
import cv2
import numpy as np
import shutil
Image.MAX_IMAGE_PIXELS = None
import torch

def label_img_match(img_path,label_path):
    for root, dirs, files in os.walk(img_path):
        start = time.time()
        img_files = [file_name for file_name in files]
    for root, dirs, files in os.walk(label_path):
        labels_files = [file_name for file_name in files]
        for i in labels_files:
            # t = cv2.imread(os.path.join(root, i), 0)
            # print(t.shape)
            # i = i.split('_')[0]+'_sat.jpg'
            if i not in img_files:
                return i
    return 'same'

def same_name(img_path):
    path_list = os.listdir(img_path)
    for i in path_list:
        img = cv2.imread(os.path.join(img_path, i), -1)
        print(img.shape)
        s = i.split('.')[0]+'.png'
        save_path_ = os.path.join(img_path, s)
        os.rename(os.path.join(img_path, i), save_path_)

# 获取图像信息
def get_data_info(dir_path):
    size = 0
    number = 0
    bad_number = 0
    for root, dirs, files in os.walk(dir_path):
        start = time.time()
        img_files = [file_name for file_name in files] # if fileName.endswith(extension) for extension in ['.jpg', '.png', '.jpeg','tif']
        files_size = sum([os.path.getsize(os.path.join(root, file_name)) for file_name in img_files])
        files_number = len(img_files)
        size += files_size
        number += files_number
        print(time.time()-start)
        for file in img_files:
            try:
                img = Image.open(os.path.join(root, file))
                img.load()
            except OSError:
                bad_number += 1
    return size / 1024 / 1024, number, bad_number

# 过滤模糊图片
def filter_blurred(dir_path):
    filter_dir = os.path.join(os.path.dirname(dir_path), 'filter_blurred')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    filter_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files]
        for file in img_files:
            file_path = os.path.join(root, file)
            # img = cv2.imread(file_path)
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)  # 从内存中的缓冲区读取图像
            image_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if image_var < 100:
                shutil.move(file_path, filter_dir)
                filter_number += 1
    return filter_number

# 计算两张图片的相似度
def calc_similarity(img1_path, img2_path):
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), -1)
    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])  # 计算图直方图
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])  # 计算图直方图
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    similarity1 = cv2.compareHist(H1, H2, 0)  # 相似度比较
    print('similarity:', similarity1)
    if similarity1 > 0.98:  # 0.98是阈值，可根据需求调整
        return True
    else:
        return False

# 去除相似度高的图片
def filter_similar(dir_path):
    filter_dir = os.path.join(os.path.dirname(dir_path), 'filter_similar')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)
    filter_number = 0
    for root, dirs, files in os.walk(dir_path):
        img_files = [file_name for file_name in files]
        filter_list = []
        for index in range(len(img_files))[:-4]:
            if img_files[index] in filter_list:
                continue
            for idx in range(len(img_files))[(index+1):(index+5)]:
                img1_path = os.path.join(root, img_files[index])
                img2_path = os.path.join(root, img_files[idx])
                if calc_similarity(img1_path, img2_path):
                    filter_list.append(img_files[idx])
                    filter_number += 1
        for item in filter_list:
            src_path = os.path.join(root, item)
            shutil.move(src_path, filter_dir)
    return filter_number

def move_labels(labels_path,similar):
    num = 0
    labels_similars = os.path.join(os.path.dirname(labels_path), 'labels_similars')
    if not os.path.exists(labels_similars):
        os.mkdir(labels_similars)
    for root, fir, files in os.walk(similar):
        img_files = [file_name for file_name in files]
    for root, fir, files in os.walk(labels_path):
        g_files = [file_name for file_name in files]
    for item in g_files:
        if item not in img_files:
            continue
        print(item)
        src_path = os.path.join(labels_path, item)
        shutil.move(src_path, labels_similars)
        num += 1
    return num
def text(path):
    img = cv2.imread(path,-1)
    print(img)


if __name__ == '__main__':
    # print(label_img_match(r'/data/chenyuxia/roads/deeplobe_filter/CHN6-CUG/all_img/images_filteres', r'/data/chenyuxia/roads/deeplobe_filter/CHN6-CUG/all_img/labels_filteres'))
    # print(get_data_info(r'/data/chenyuxia/new/images'))
    # print(filter_blurred(r'/data/chenyuxia/new/images'))
    # print(filter_similar(r'/data/chenyuxia/new/images'))
    # print(move_labels(r'/data/chenyuxia/new/labels', r'/data/chenyuxia/new/filter_similar'))
    # print(same_name(r'/data/chenyuxia/roads/deeplobe_filter/CHN6-CUG/all_img/mask'))
    print(text(r'/data/chenyuxia/roads/deeplobe_filter/all_roads/deep/train/labels/0_0_0_0_122555_sat_256_256_256_256.png'))