import numpy as np
from PIL import Image
import argparse
import os
import os.path as osp
import glob
import cv2

from shutil import copy


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--from_dir', default=r'C:\Users\Administrator\Documents\金山区\images')
    parser.add_argument('--source_dir', default=r'C:\Users\Administrator\Documents\金山区\labels')
    parser.add_argument('--to_dir', default=r'C:\Users\Administrator\Documents\金山区清洗\images')
    parser.add_argument('--to_dir_label', default=r'C:\Users\Administrator\Documents\金山区清洗\labels')

    # parser.add_argument('--type', help='string, which process do you want')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 初始化参数
    from_dir = args.from_dir
    to_dir = args.to_dir
    to_dir_label = args.to_dir_label
    source_dir = args.source_dir
    # 判断路径否存在
    if not osp.exists(to_dir):
        os.makedirs(to_dir)
        os.makedirs(to_dir_label)
        print('Creating train data directory:', to_dir, to_dir_label)

    # JL1KF01A__05

    for index, file_name in enumerate(glob.glob(osp.join(from_dir, '*.png'))):
        base = osp.splitext(osp.basename(file_name))[0] + '.png'
        from_label = os.path.join(source_dir, base)  # 旧文件的绝对路径(包含文件的后缀名)
        to_dir0 = os.path.join(to_dir, base)
        to_dir_label0 = os.path.join(to_dir_label, base)

        try:
            image = Image.open(from_label)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('\n' + message)
            os.remove(file_name)
            os.remove(from_path)
            continue

        image_n = np.array(image)
        # 不止一类标签 & 如果只有一类，只能是255
        if len(np.unique(image_n)) != 1 or np.unique(image_n)[0] == 255:
            print('copy', file_name, 'to', to_dir0)
            copy(file_name, to_dir0)
            copy(from_label, to_dir_label0)
