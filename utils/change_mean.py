import cv2
import numpy as np
import os
def change_mean(img_path,save):
    for root, dirs, files in os.walk(img_path):
        img_files = [file_name for file_name in files]
        for file in img_files:
            # 读取图像
            image = cv2.imread(os.path.join(img_path, file), -1)
            # 获取图像高，宽，深度
            h, w, d = image.shape
            # 添加数组，高宽与原图像相等
            add_bimg = np.ones((h, w), dtype=np.uint8) * 40
            # opencv里自带了分离三通道的函数split()，返回值依次是蓝色、绿色和红色通道的灰度图
            # 可以直接显示BGR图像
            b, g, r = cv2.split(image)
            # 将波段加数组
            b += add_bimg
            # 存储图
            merges = cv2.merge([b,g,r])
            cv2.imwrite(os.path.join(save, file + 't.tif'), merges)

if __name__ == '__main__':
    print(change_mean('/data/chenyuxia/tianjin/taibei', '/data/chenyuxia/taibei_text'))