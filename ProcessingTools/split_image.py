import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import math
from PIL import Image


def split_image(path):
    for i in os.listdir(path):
        pic_path = os.path.join(path, i)
        img = cv2.imread(pic_path)
        # img = Image.open(pic_path)
        print(img.shape)
        (height, width, _) = img.shape
        height_list = [math.ceil(height / 2), height - math.ceil(height / 2)]
        width_list = [math.ceil(width / 2), width - math.ceil(width / 2)]

        cur_h, pre_h = 0, 0
        k = 0
        for i in range(len(height_list)):
            h = height_list[i]
            pre_h = cur_h
            cur_h += h
            cur_w, pre_w = 0, 0
            for j in range(len(width_list)):
                w = width_list[j]
                pre_w = cur_w
                cur_w += w
                print(pre_h, cur_h, pre_w, cur_w)
                img_ = img[pre_h:cur_h, pre_w:cur_w]
                print(img_.shape)
                cv2.imwrite('/data/fpc/inference/taibei_splited/taibei_splited_{}.tif'.format(k), img_)
                k += 1


def images_merge(file):
    pic_list = []
    for i in os.listdir(file):
        name = os.path.join(file, i)
        img = cv2.imread(name, -1)
        # print(img.shape)
        pic_list.append(img)

    file_length = len(pic_list)
    height = width = int(pow(file_length, 1/2))
    # print(height)

    initial_h = 0
    blank_switch = True
    for h in range(height):
        initial_w = pic_list[h * height]
        for w in range(width - 1):
            # print('w', w + h * height + 1)
            initial_w = np.concatenate((initial_w, pic_list[w + 1 + h * height]), axis=1)  # axis=0 按垂直方向，axis=1 按水平方向
        if blank_switch:
            initial_h = initial_w
            blank_switch = False
        else:
            initial_h = np.concatenate((initial_h, initial_w), axis=0)
            print(initial_h.shape)

    print(initial_h.shape)

    cv2.imwrite('/data/fpc/output/res/res32.png', initial_h)


if __name__ == "__main__":
    # split_image('/data/fpc/inference/taibei/')
    images_merge('/data/fpc/output/outputs_10_21_3/maskOnRes')