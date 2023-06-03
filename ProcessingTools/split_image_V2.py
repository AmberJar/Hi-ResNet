# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


def splitimage(src, rownum, colnum, dstpath):
    # cv2读 hwc
    img = cv2.imread(src, 1)
    h, w, _ = img.shape
    # img = Image.open(src)
    # w, h = img.size

    if rownum <= h and colnum <= w:
        # print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')

        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = str(fn[0])

        # ext = fn[-1]
        ext = '.png'

        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                # 左上xy 右下xy
                # box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                # img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num).zfill(3) + '.' + ext), ext)
                #
                _img = img[c * colwidth:(c + 1) * colwidth, r * rowheight:(r + 1) * rowheight]
                save_path_img = os.path.join(str(dstpath), basename + '_' + str(str(num).zfill(3)) + ext)
                cv2.imwrite(save_path_img, _img)
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')


def splited_pic2origin_pic(src, rownum, colnum, dstpath):
    pic_list = os.listdir(src)
    col_list = []
    for i in range(rownum):
        row_list = []
        for j in range(colnum):
            cur_pic = pic_list[j + i * colnum]
            cur_pic_path = os.path.join(src, cur_pic)
            img = cv2.imread(cur_pic_path)
            row_list.append(img)
        row = np.concatenate(row_list, axis=0)
        col_list.append(row)
    res = np.concatenate(col_list, axis=1)
    cv2.imwrite(os.path.join(dstpath, 'merged_mask_plants_001.png'), res)



if __name__ == '__main__':
    # 分离part
    # src = '/data/fpc/inference/taian_data/gamma_correct_pic/all_gamma_correct.tif'
    # # for i in range(4):
    # #     src = '/data/fpc/inference/taibei_splited_balanced/taibei_splited_{}.tif'.format(i)
    #
    # if os.path.isfile(src):
    #     dstpath = '/data/fpc/inference/taian_data/taian_splited_gamma_6png'
    #     if not os.path.exists(dstpath):
    #         os.mkdir(dstpath)
    #     if (dstpath == '') or os.path.exists(dstpath):
    #         row = 3
    #         col = 2
    #         if row > 0 and col > 0:
    #             splitimage(src, row, col, dstpath)
    #         else:
    #             print('无效的行列切割参数！')
    #     else:
    #         print('图片输出目录 %s 不存在！' % dstpath)
    # else:
    #     print('图片文件 %s 不存在！' % src)

    # 拼接part
    src = '/data/fpc/output/outputs_12_18_vege_01'
    dst = '/data/fpc/inference/taian_data/tmp'
    row, column = 3, 2
    splited_pic2origin_pic(src, row, column, dst)