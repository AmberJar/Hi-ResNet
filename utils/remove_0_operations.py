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
    parser.add_argument('--from_image_dir', default=r'C:\Users\Administrator\Documents\松江区东\images')
    parser.add_argument('--from_label_dir', default=r'C:\Users\Administrator\Documents\松江区东\labels')
    parser.add_argument('--to_image_dir', default=r'C:\Users\Administrator\Documents\松江区\images')
    parser.add_argument('--to_label_dir', default=r'C:\Users\Administrator\Documents\松江区\labels')
    # parser.add_argument('--type', help='string, which process do you want')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 初始化参数
    from_image_dir = args.from_image_dir
    from_label_dir = args.from_label_dir
    to_image_dir = args.to_image_dir
    to_label_dir = args.to_label_dir
    # 判断路径否存在
    if not osp.exists(to_label_dir):
        os.makedirs(to_label_dir)
        os.makedirs(to_image_dir)
        print('Creating train data directory:', to_image_dir)

    # JL1KF01A__05

    for index, file_name in enumerate(glob.glob(osp.join(from_image_dir, '*.png'))):
        base = osp.splitext(osp.basename(file_name))[0] + '.png'
        from_path = os.path.join(from_image_dir, base)  # 旧文件的绝对路径(包含文件的后缀名)

        try:
            image = Image.open(file_name).convert(mode='L')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('\n' + message)
            continue
        image = np.array(image)

        shape = image.shape
        if np.unique(image)[0] == 0 and len(np.unique(image)) == 1:
            label_path = os.path.join(from_label_dir, base)  # 新文件的绝对路径
            # 处理标签
            print('remove file',file_name)
            os.remove(file_name)
            os.remove(label_path)


