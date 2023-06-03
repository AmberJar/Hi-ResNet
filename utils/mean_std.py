###https://blog.csdn.net/weixin_43105540/article/details/119570461
from itertools import repeat
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

NUM_THREADS = os.cpu_count()


def calc_channel_sum(img_path):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0  # 准换为RGB的array形式
    h, w, _ = img.shape
    pixel_num = h * w
    channel_sum = img.sum(axis=(0, 1))  # 各颜色通道像素求和
    return channel_sum, pixel_num


def calc_channel_var(img_path, mean):  # 计算标准差的辅助函数
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0
    channel_var = np.sum((img - mean) ** 2, axis=(0, 1))
    return channel_var


if __name__ == '__main__':
    train_path = Path(r'/data/liyonggang/adaspace/val/images')
    img_f = list(train_path.rglob('*.png'))
    n = len(img_f)
    result = ThreadPool(NUM_THREADS).imap(calc_channel_sum, img_f)  # 多线程计算
    channel_sum = np.zeros(3)
    cnt = 0
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        channel_sum += x[0]
        cnt += x[1]
    mean = channel_sum / cnt
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (mean[0], mean[1], mean[2]))

    result = ThreadPool(NUM_THREADS).imap(lambda x: calc_channel_var(*x), zip(img_f, repeat(mean)))
    channel_sum = np.zeros(3)
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        channel_sum += x
    var = np.sqrt(channel_sum / cnt)
    print("R_var is %f, G_var is %f, B_var is %f" % (var[0], var[1], var[2]))