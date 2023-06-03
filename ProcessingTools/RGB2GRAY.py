import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

# 读入的图像是BGR空间图像
label_path = '/data/fpc/data/archive/0.5/labels'
save_path = '/data/fpc/data/archive/0.5/labels_'


def pic_read(label_path, save_path):
    path_list = os.listdir(label_path)

    for i in path_list:
        # print(i)
        label = os.path.join(label_path, i)
        GRAY = RGB2GRAY(label)
        save_path_ = os.path.join(save_path, i)
        cv2.imwrite(save_path_, GRAY)


def RGB2GRAY(label):
    img = cv2.imread(label)
    b, g, r = cv2.split(img)
    # print(b, g, r)
    r = r * 0.1
    g = g * 0.25
    b = b * 0.6

    res = np.ceil(r + g + b)
    # print(res[0][0])
    res[res == urban] = 0
    res[res == agriculture] = 1
    res[res == random] = 2
    res[res == forest] = 3
    res[res == water] = 4
    res[res == barred] = 5
    res[res == back] = 6

    return res


def RGB_Calculation(rgb_list):
    r = rgb_list[0]
    g = rgb_list[1]
    b = rgb_list[2]
    res = math.ceil(r * 0.1 + g * 0.25 + b * 0.6)

    return res


# pre-defined
# 定义不同mask的颜色 RGB
urban_land = [0, 255, 255] #0
agriculture_land = [255, 255, 0] #1
rangeland = [255, 0, 255] #2
forest_land = [0, 255, 0] #3
water = [0, 0, 255] #4
barren_land = [255, 255, 255] #5
background = [0, 0, 0] #6

urban = RGB_Calculation(urban_land)
agriculture = RGB_Calculation(agriculture_land)
random = RGB_Calculation(rangeland)
forest = RGB_Calculation(forest_land)
water = RGB_Calculation(water)
barred = RGB_Calculation(barren_land)
back = RGB_Calculation(background)

print(urban)
print(agriculture)
print(random)
print(forest)
print(water)
print(barred)
print(back)

if __name__ == "__main__":
    time1 = time.time()

    pic_read(label_path, save_path)

    time2 = time.time()
    print(time2 - time1)
