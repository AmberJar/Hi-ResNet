import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg


def union_image_mask(image_path, mask_path, num):
    # 读取原图
    image = cv2.imread(image_path)
    print(image.shape)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 裁剪到和原图一样大小
    print(mask_2d.shape)
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)
    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    cv2.imwrite(os.path.join("/data/chenyuxia/outputs/res_roads", str(num) + ".bmp"), image)


def mask2source(image_path, mask_path, output_path, alpha):
    # root_path_background = "/home/cc/codes/python/YOLOP-main/datas/images/train/"
    # root_path_paste = "/home/cc/codes/python/YOLOP-main/datas/da_seg_annotations/train/"
    # output_path = "/home/cc/codes/python/YOLOP-main/datas/masks/"
    # img_list = os.listdir(root_path_background)
    # label_list = os.listdir(root_path_paste)
    # img_list = sorted(img_list)
    # label_list = sorted(label_list)
    # for num, img_label in enumerate(zip(img_list, label_list)):
    img = cv2.imread(image_path)
    label = cv2.imread(mask_path, 0)
    # label[(label.any() != 3 and label.any() != 1)] = 0
    # label[label == 1] = 60
    # label[label == 3] = 180
    pd = np.zeros(img.shape)
    pd[np.where(label == 0)] = [0, 0, 0] #黑
    pd[np.where(label == 1)] = [0, 0, 0]  # 绿
    pd[np.where(label == 2)] = [184, 134, 11]  # 棕色
    pd[np.where(label == 3)] = [0, 0, 0] # Mediumblue
    pd[np.where(label == 4)] = [0, 0, 0] # deep pink
    pd[np.where(label == 5)] = [0, 0, 0]  # Firebrick
    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    img_merge = alpha * img + (1 - alpha) * pd
    cv2.imwrite(output_path, img_merge)


if __name__ == '__main__':
    mask_pth = '/data/chenyuxia/outputs/outputs_10_17'
    img_path = '/data/chenyuxia/16_taibei/taibei'
    out_path = '/data/chenyuxia/outputs/outputs_10_17'
    for i in os.listdir(img_path):
        input_image = os.path.join(img_path, i)
        j = i.split('.')[0] + '_mask' + '.png'
        print(j)
        input_mask = os.path.join(mask_pth, j)
        output = os.path.join(out_path, i)
        mask2source(input_image, input_mask, output, 0.6)
