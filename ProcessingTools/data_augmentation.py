# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy
from tqdm import tqdm
import random
import math
from PIL import ImageEnhance
from PIL import Image


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


class Shift:
    def __init__(self, limit=50, prob=1.0):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                mask1 = cv2.copyMakeBorder(mask, limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
                mask = mask1[y1:y2, x1:x2]

        return img, mask


class Cutout:
    def __init__(self, num_holes=3, max_h_size=50, max_w_size=50, fill_value=0, prob=1.):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            h = img.shape[0]
            w = img.shape[1]
            # c = img.shape[2]
            # img2 = np.ones([h, w], np.float32)
            for _ in range(self.num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
                y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
                x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
                x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
                img[y1: y2, x1: x2, :] = self.fill_value
                if mask is not None:
                    mask[y1: y2, x1: x2] = self.fill_value
        return img, mask


class Rescale(object):
    def __init__(self, output_size=320, prob=0.75):
        self.prob = prob
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, image, label):
        if random.random() < self.prob:
            raw_h, raw_w = image.shape[:2]

            img = cv2.resize(image, (self.output_size, self.output_size))
            lbl = cv2.resize(label, (self.output_size, self.output_size))

            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                img = img[i:i + raw_h, j:j + raw_h]
                lbl = lbl[i:i + raw_h, j:j + raw_h]
            else:
                res_h = raw_w - h
                img = cv2.copyMakeBorder(img, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
                lbl = cv2.copyMakeBorder(lbl, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
            return img, lbl
        else:
            return image, label


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.02, sh=0.4, r1=0.3, mean=(0.356133, 0.343815, 0.376904)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, mask):
        if random.uniform(0, 1) >= self.probability:
            return img, mask
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness, color=ImageEnhance.Color
)


class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i] * 2.0 - 1.0) + 1  # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)

        return out


class DualCompose:
    def __init__(self, transforms, x, mask):
        self.transforms = transforms
        self.x = x
        self.mask = mask

    def __call__(self):
        for i, t in enumerate(self.transforms):
            self.x, self.mask = t(self.x, self.mask)

        return self.x, self.mask


if __name__ == '__main__':
    # 图片文件夹路径
    img_dir = r'/data/fpc/data/love_DA/loveDA/train/images_filtered_90'
    mask_dir = r'/data/fpc/data/love_DA/loveDA/train/labels_filtered_90'
    img_dir_save = r'/data/fpc/data/love_DA/loveDA/train/images_'
    mask_dir_save = r'/data/fpc/data/love_DA/loveDA/train/labels_'

    if not os.path.exists(img_dir_save):
        os.mkdir(img_dir_save)
    if not os.path.exists(mask_dir_save):
        os.mkdir(mask_dir_save)

    # mode list
    # flip, rotate90, rotate180, rotate270, gauss
    mode_list = ['flip', 'rotate90', 'rotate180', 'rotate270', 'gaussian', 'cut_out', 'rescale', 'shift', 'jitter']
    mode_pool = ['flip', 'shift', 'rotate90', 'rotate180', 'rotate270']

    for img_name in tqdm(os.listdir(img_dir)):
        num = random.randint(0, len(mode_pool) - 1)
        mode = mode_pool[num]
        assert mode in mode_list

        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # class_list = np.unique(mask)
        # aug_list = [7, 8, 5, 14, 10, 11, 12, 15]
        # intersect = list(set(class_list) & set(aug_list))
        #
        # if not intersect:
        #     continue

        if mode == 'rotate90':
            # 旋转90
            img_res = rotate(img, 90)
            mask_res = rotate(mask, 90)

            img_save_path = os.path.join(img_dir_save, 'rot90_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'rot90_' + img_name)
        elif mode == 'rotate180':
            # 旋转180
            img_res = rotate(img, 180)
            mask_res = rotate(mask, 180)

            img_save_path = os.path.join(img_dir_save, 'rot180_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'rot180_' + img_name)
        elif mode == 'rotate270':
            # 旋转270
            img_res = rotate(img, 270)
            mask_res = rotate(mask, 270)

            img_save_path = os.path.join(img_dir_save, 'rot270_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'rot270_' + img_name)
        elif mode == 'flip':
            # 镜像
            img_res = flip(img)
            mask_res = flip(mask)

            img_save_path = os.path.join(img_dir_save, 'flip_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'flip_' + img_name)
        elif mode == 'gaussian':
            # 增加噪声
            # img_salt = SaltAndPepper(img, 0.3)
            # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)

            img_res = addGaussianNoise(img, 0.3)
            mask_res = mask
            img_save_path = os.path.join(img_dir_save, 'gauss_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'gauss_' + img_name)
        elif mode == 'cut_out':
            transform = DualCompose([
                # Rescale(),
                Cutout(),
                # Shift(),
            ], img, mask)
            img_res, mask_res = transform()
            img_save_path = os.path.join(img_dir_save, 'cutout_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'cutout_' + img_name)
        elif mode == 'rescale':
            transform = DualCompose([
                Rescale(),
                # Cutout(),
                # Shift(),
            ], img, mask)
            img_res, mask_res = transform()
            img_save_path = os.path.join(img_dir_save, 'rescale_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'rescale_' + img_name)
        elif mode == 'shift':
            transform = DualCompose([
                # Rescale(),
                # Cutout(),
                Shift(),
            ], img, mask)
            img_res, mask_res = transform()
            img_save_path = os.path.join(img_dir_save, 'shift_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'shift_' + img_name)
        elif mode == 'jitter':
            _transform_dict = {'brightness': 0.1026, 'contrast': 0.0935, 'sharpness': 0.8386, 'color': 0.1592}
            _color_jitter = ColorJitter(_transform_dict)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = _color_jitter(img)
            img_res = np.array(img)
            mask_res = mask
            img_save_path = os.path.join(img_dir_save, 'jitter_' + img_name)
            mask_save_path = os.path.join(mask_dir_save, 'jitter_' + img_name)

        # 存，记得改名字
        cv2.imwrite(img_save_path, img_res)
        cv2.imwrite(mask_save_path, mask_res)

        # # 变亮、变暗
        # img_darker = darker(img)
        # cv2.imwrite(file_dir + img_name[0:-4] + '_darker.jpg', img_darker)
        # img_brighter = brighter(img)
        # cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.jpg', img_brighter)
        #
        # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
        # #      cv2.GaussianBlur(图像，卷积核，标准差）
        # cv2.imwrite(file_dir + img_name[0:-4] + '_blur.jpg', blur)
