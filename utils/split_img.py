import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import torch
from tqdm import tqdm
import math
from PIL import Image


def get_img_info(img,part = 4):
    h, w, _ = img.shape
    h_step = int(np.ceil(h / part))
    w_step = int(np.ceil(w / part))
    n_h = (h_step) * part
    n_w = (w_step) * part
    h_fill = n_h - h
    w_fill = n_w - w
    # image = np.asarray(img.convert('RGB'),
    #                    dtype=np.float32)
    image = torch.from_numpy(np.asarray(img, dtype=np.float32))
    return h, w, n_h, n_w, h_step, w_step, h_fill, w_fill

def img_fill_split(img_path, output_path, part = 4):
    img = cv2.imread(img_path, -1)
    img_infos = []
    num = 0
    h, w, n_h, n_w, h_step, w_step, h_fill, w_fill = get_img_info(img, part)
    print(get_img_info(img,part))
    #img_fill = cv2.copyMakeBorder(img, 0, h_fill, 0, w_fill, cv2.BORDER_CONSTANT)
    for i in tqdm(np.arange(0, n_h, h_step)):
        for j in np.arange(0, n_w, w_step):
            if i + h_step >= h:
                h_i = h
            else:
                h_i = i + h_step
            if i + w_step >= w:
                w_j = w
            else:
                w_j = j + w_step

            img_split = img[i : h_i,j : w_j]
            img_split_nam = 'split_' + str(num) + '.tif'
            num += 1
            cv2.imwrite(os.path.join(output_path, img_split_nam), img_split)

    return n_h



def split_image(path):
    z = 0
    for i in os.listdir(path):
        print(i)
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
                print(pre_h, cur_h - 1, pre_w, cur_w - 1)
                img_ = img[pre_h:cur_h - 1, pre_w:cur_w - 1]
                cv2.imwrite('/data/chenyuxia/outputs/res_roads/text/taibei_splited_{}-{}.png'.format(z, k), img_)
                k += 1
        z += 1

def images_merge(file):
    pic_list = []
    for i in os.listdir(file):
        name = os.path.join(file, i)
        img = cv2.imread(name)
        pic_list.append(img)

    img_x = np.concatenate((pic_list[0], pic_list[1]), axis=1)  # axis=0 按垂直方向，axis=1 按水平方向
    img_y = np.concatenate((pic_list[2], pic_list[3]), axis=1)
    res = np.concatenate((img_x, img_y), axis=0)
    cv2.imwrite('/data/chenyuxia/10_outputs/res.tif', res)


if __name__ == "__main__":
    print(split_image(r'/data/chenyuxia/outputs/res_roads'))
    # images_merge(r'/data/chenyuxia/10_outputs')
    # img = cv2.imread(r'/data/chenyuxia/taibei_splited/split_10.tif',-1)
    # print(img.shape)
    # print(img_fill_split('/data/chenyuxia/taibei/taibei_splited_1.tif','/data/chenyuxia/taibei_splited',4))