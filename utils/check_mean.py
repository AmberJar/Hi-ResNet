# from torchvision.transforms import ToTensor#用于把图片转化为张量
# import numpy as np#用于将张量转化为数组，进行除法
# from torchvision.datasets import ImageFolder#用于导入图片数据集
# from PIL import ImageFile
# from PIL import Image
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None
# means = [0,0,0]
# std = [0,0,0]#初始化均值和方差
# transform = ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
# dataset = ImageFolder(r"/data/chenyuxia/roads/deeplobe_filter/CHN6-CUG/train/", transform=transform)#导入数据集的图片，并且转化为张量
# num_imgs=len(dataset)#获取数据集的图片数量
# for img, a in dataset:#遍历数据集的张量和标签
#     for i in range(3):#遍历图片的RGB三通道
#         # 计算每一个通道的均值和标准差
#         means[i] += img[i, :, :].mean()
#         std[i] += img[i, :, :].std()
# mean = np.array(means)/num_imgs
# std = np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
# print(mean, std)#打印出结果
# print(mean*255, std*255)
import cv2
import os
import numpy as np
import math
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集

path = '/data/fpc/NAIC/balance_11_7w/train/images'

b_list, g_list, r_list = [], [], []
bs_list, gs_list, rs_list = [], [], []

for i in os.listdir(path):
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
