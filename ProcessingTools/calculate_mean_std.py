import sys

import cv2
import os
import numpy as np
import math
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集
from tqdm import tqdm
import torch

path = '/mnt/data/chenyuxia/EXPERIMENTAL/loveda/final_trainset/train/images'

b_list, g_list, r_list = [], [], []
bs_list, gs_list, rs_list = [], [], []

device = 'cpu'

if device == 'cpu':
    for i in tqdm(os.listdir(path)):
        pic = os.path.join(path, i)
        img = cv2.imread(pic).astype(np.float64)

        b, g, r = cv2.split(img)

        b /= 255.
        g /= 255.
        r /= 255.

        b_list.append(np.mean(b))
        g_list.append(np.mean(g))
        r_list.append(np.mean(r))

        bs_list.append(np.std(b))
        gs_list.append(np.std(g))
        rs_list.append(np.std(r))

    print([np.mean(r_list).round(6), np.mean(b_list).round(6), np.mean(g_list).round(6)])
    print([np.mean(rs_list).round(6), np.mean(bs_list).round(6), np.mean(gs_list).round(6)])
elif device == 'gpu':
    res = None
    k = 0
    dir_length = len(os.listdir(path))
    for i in tqdm(os.listdir(path)):
        pic = os.path.join(path, i)
        img = cv2.imread(pic).astype(np.float64)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).to(torch.cuda.current_device())
        img = torch.mean(img, dim=(1,2)).unsqueeze(0)

        if k == 0: res = img
        else: res = torch.cat((res, img), 0)
        k += 1

    print(torch.mean(res, dim=0))



