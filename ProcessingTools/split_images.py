# 切图片
import cv2
import math
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

file_path = "/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/val/images/"  # 原图路径
gt_path = "/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/val/labels/"  # mask路径
save_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/val/images_256/'  # 原图crop后保存路径
save_gt_path = '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/raw/val/labels_256/'  # mask crop后保存路径
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(save_gt_path).mkdir(parents=True, exist_ok=True)

for filename in tqdm(os.listdir(file_path)):
    # crop的size
    height = 256
    width = 256

    # overlap大小
    over_x = 0
    over_y = 0
    h_val = height - over_x
    w_val = width - over_y

    # 设置是否丢弃不符合大小的图像
    mandatory = True

    img = cv2.imread(file_path + filename)  # 原图按照3通道彩色读入
    # print(file_path + filename)
    gt_filename = filename.split('.')[0] + '.png'
    gt_img = cv2.imread(gt_path + gt_filename, 0)  # mask按照灰度图单通道读入
    # print("img info :", filename, img.shape)
    # print("gt info :", filename, gt_img.shape)

    # original image size
    original_height = img.shape[0]
    original_width = img.shape[1]

    max_row = float((original_height - height) / h_val) + 1
    max_col = float((original_width - width) / w_val) + 1

    max_row = math.ceil(max_row) if mandatory == False else math.floor(max_row)
    max_col = math.ceil(max_col) if mandatory == False else math.floor(max_col)

    images = []
    images_gt = []
    name = filename.split('.')[0]


    for i in range(max_row):
        images_temp = []
        images_temp_gt = []
        for j in range(max_col):
            temp_path = save_path + '/' + str(i) + '_' + str(j) + '_'
            temp_gt_path = save_gt_path + '/' + str(i) + '_' + str(j) + '_'
            # 判断是否超出边界
            if ((width + j * w_val) > original_width and (
                    i * h_val + height) <= original_height):  # Judge the right most incomplete part
                temp = img[i * h_val:i * h_val + height, j * w_val:original_width, :]
                temp_gt = gt_img[i * h_val:i * h_val + height, j * w_val:original_width]

                w_padding = int((width + j * w_val) - original_width)
                temp_path = temp_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'
                temp_gt_path = temp_gt_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'

                # 对超出边界部分进行填充，我这里是对边缘像素进行镜像
                temp = cv2.copyMakeBorder(temp, 0, 0, 0, w_padding, cv2.BORDER_REFLECT)
                temp_gt = cv2.copyMakeBorder(temp_gt, 0, 0, 0, w_padding, cv2.BORDER_REFLECT)

                cv2.imwrite(temp_path, temp)
                cv2.imwrite(temp_gt_path, temp_gt)
                images_temp.append(temp)
                images_temp_gt.append(temp_gt)
            elif ((height + i * h_val) > original_height and (
                    j * w_val + width) <= original_width):  # Judge the incomplete part at the bottom

                h_padding = int((height + i * h_val) - original_height)
                # w_padding = int((width + j * w_val) - original_width)
                temp = img[i * h_val:original_height, j * w_val:j * w_val + width, :]
                temp_gt = gt_img[i * h_val:original_height, j * w_val:j * w_val + width]

                temp_path = temp_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'
                temp_gt_path = temp_gt_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'

                temp = cv2.copyMakeBorder(temp, 0, h_padding, 0, 0, cv2.BORDER_REFLECT)
                temp_gt = cv2.copyMakeBorder(temp_gt, 0, h_padding, 0, 0, cv2.BORDER_REFLECT)

                cv2.imwrite(temp_path, temp)
                cv2.imwrite(temp_gt_path, temp_gt)
                images_temp.append(temp)
                images_temp_gt.append(temp_gt)
            elif ((width + j * w_val) > original_width and (
                    i * h_val + height) > original_height):  # Judge the last slide
                h_padding = int((height + i * h_val) - original_height)
                w_padding = int((width + j * w_val) - original_width)

                temp = img[i * h_val:original_height, j * w_val:original_width, :]
                temp_gt = gt_img[i * h_val:original_height, j * w_val:original_width]

                temp_path = temp_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'
                temp_gt_path = temp_gt_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'

                temp = cv2.copyMakeBorder(temp, 0, h_padding, 0, w_padding, cv2.BORDER_REFLECT)
                temp_gt = cv2.copyMakeBorder(temp_gt, 0, h_padding, 0, w_padding, cv2.BORDER_REFLECT)

                cv2.imwrite(temp_path, temp)
                cv2.imwrite(temp_gt_path, temp_gt)
                images_temp.append(temp)
                images_temp_gt.append(temp_gt)
            else:
                temp = img[i * h_val:i * h_val + height, j * w_val:j * w_val + width, :]
                temp_gt = gt_img[i * h_val:i * h_val + height, j * w_val:j * w_val + width]

                temp_path = temp_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'
                temp_gt_path = temp_gt_path + name + '_' + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.png'

                cv2.imwrite(temp_path, temp)
                cv2.imwrite(temp_gt_path, temp_gt)
                images_temp.append(temp)
                images_temp_gt.append(temp_gt)


        images.append(images_temp)
        images_gt.append(images_temp_gt)

    # print(len(images))