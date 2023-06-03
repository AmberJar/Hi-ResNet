import numpy as np
from PIL import Image
import argparse
import os
import os.path as osp
import glob
import cv2

from shutil import copy
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--from_image_dir', default=r'/data/liyonggang/adaspace_val/val/images')
    parser.add_argument('--from_label_dir', default=r'/data/liyonggang/adaspace_val/val/labels')
    parser.add_argument('--to_image_dir', default=r'/data/liyonggang/adaspace_val/train/images')
    parser.add_argument('--to_label_dir', default=r'/data/liyonggang/adaspace_val/train/labels')
    # parser.add_argument('--type', help='string, which process do you want')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 初始化参数
    from_image_dir = args.from_image_dir
    from_label_dir = args.from_label_dir
    to_image_dir = args.to_image_dir
    to_label_dir = args.to_label_dir

    for index, file_name in enumerate(glob.glob(osp.join(from_image_dir, '*.png'))):
        if index % 4 == 0:
            base = osp.splitext(osp.basename(file_name))[0] + '.png'
            to_image_dir0 = os.path.join(to_image_dir, base)
            from_label_dir0 = os.path.join(from_label_dir, base)
            to_label_dir0 = os.path.join(to_label_dir, base)
            shutil.move(file_name, to_image_dir0)
            shutil.move(from_label_dir0, to_label_dir0)
        #
        # try:
        #     image = Image.open(file_name).convert(mode='L')
        # except Exception as ex:
        #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #     message = template.format(type(ex).__name__, ex.args)
        #     print('\n' + message)
        #     continue
        # image = np.array(image)

        # shape = image.shape
        # print(shape, file_name)
        # if np.unique(image)[0] == 255:
        #     to_path = os.path.join(to_dir, base)  # 新文件的绝对路径
        #     # 处理标签
        #     image[::] = 0
        #     Image.fromarray(image).save(file_name)
            # os.remove(to_path)
            # os.remove(from_path)


