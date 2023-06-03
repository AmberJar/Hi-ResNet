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
    parser.add_argument('--from_dir', default=r'/data/liyonggang/water/val/labels', help='input train data directory')
    parser.add_argument('--to_dir', default=r'C:\Users\Administrator\Documents\成都\labels00', help='output train data directory')
    # parser.add_argument('--type', help='string, which process do you want')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 初始化参数
    from_dir = args.from_dir
    to_dir = args.to_dir
    # 判断路径否存在
    # if not osp.exists(to_dir):
    #     os.makedirs(to_dir)
    #     print('Creating train data directory:', to_dir)

    # JL1KF01A__05

    for index, file_name in enumerate(glob.glob(osp.join(from_dir, '*.png'))):
        base = osp.splitext(osp.basename(file_name))[0] + '.png'
        from_path = os.path.join(from_dir, base)  # 旧文件的绝对路径(包含文件的后缀名)

        try:
            image = Image.open(file_name).convert(mode='L')
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('\n' + message)
            continue
        image = np.array(image)

        shape = image.shape
        print(shape, file_name)
        if np.unique(image)[0] == 255:
            to_path = os.path.join(to_dir, base)  # 新文件的绝对路径
            # 处理标签
            image[::] = 0
            Image.fromarray(image).save(file_name)
            # os.remove(to_path)
            # os.remove(from_path)


