"""
对于栅格数据的后处理

"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
from tqdm import tqdm


def fill_drop(ori_img, area_size, mode='fill'):
    """
    填空洞和去小点
    :param ori_img:输入图像
    :param area_size:填或去的最大面积
    :param mode: 'fill'为填空洞'drop'为去小点
    :return:
    """
    ori_img.astype(np.uint8)
    if mode == 'fill':
        background = (255 - ori_img)
    elif mode == 'drop':
        background = ori_img
    background=background.astype(np.uint8)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background, connectivity=8)
    index_small_mapping = {0: 0}

    if mode == 'fill':
        for i in range(1, num_labels):
            area = stats[i][-1]
            if area > area_size:
                index_small_mapping[i] = 255
            else:
                index_small_mapping[i] = 0
    elif mode == 'drop':
        for i in range(1, num_labels):
            area = stats[i][-1]
            if area > area_size:
                index_small_mapping[i] = 0
            else:
                index_small_mapping[i] = 255

    k = np.array(list(index_small_mapping.keys()))
    v = np.array(list(index_small_mapping.values()))
    mapping_ar = np.zeros(max(k) + 1, dtype=v.dtype)  # k,v from approach #1
    mapping_ar[k] = v
    out = mapping_ar[labels]
    if mode == 'fill':
        ori_img = ori_img + out
        # ori_img = 255 - ori_img
    elif mode == 'drop':
        ori_img = ori_img - out
    return ori_img

    # # 连通域分析
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background, connectivity=8)
    #
    # background_big = np.zeros_like((background))
    # # background_big = np.zeros_like((background))
    # # background_bigger = np.zeros_like((background))
    # # 遍历联通域
    # for i in range(1, num_labels):
    #     #     print(i, num_labels)
    #     img = np.zeros_like(labels)
    #     area = np.sum(labels == i)
    #
    #     index = np.where(labels == i)
    #     img[index] = 255
    #     img = np.array(img, dtype=np.uint8)
    #     if area > 100:
    #         background_big += img


if __name__ == '__main__':
    mask_path = r'/data/fpc/output/res/res25.png'
    mask = cv2.imread(mask_path, 0)
    # fill_hole = fill_drop(mask, 50, mode='fill')
    drop_small = fill_drop(mask, 144, mode='drop')

    output_path = r'/data/fpc/output/res/res25_n.png'
    cv2.imwrite(output_path, drop_small)

