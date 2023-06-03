"""
对于栅格数据的后处理

"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
from tqdm import tqdm


def kernel_angel(kernel_num=21, l=5):
    kernel = np.ones([kernel_num, kernel_num])
    for i in range(l):
        for j in range(l - i):
            kernel[i, j] = 0
            kernel[kernel_num - 1 - i, kernel_num - j - 1] = 0
            kernel[i, kernel_num - j - 1] = 0
            kernel[kernel_num - 1 - i, j] = 0
    return kernel


def drop_block(image, drop_size=20):
    # 连通域分析

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    background_small = np.zeros_like(image)

    # 遍历联通域

    for i in tqdm(range(1, num_labels)):

        img = np.zeros_like(labels)

        area = np.sum(labels == i)

        index = np.where(labels == i)
        img[index] = 255
        img = np.array(img, dtype=np.uint8)
        if area > drop_size:
            background_small += img
    return background_small


def dilation_erosion(img, kernel_size=10, dst=10, model='dilation'):
    """
    膨胀,侵蚀
    :param model: 'dilation'膨胀，'erosion'侵蚀
    :return:
    """
    img =img.astype('uint8')
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if model == 'dilation':
        image = cv2.dilate(img, kernel, dst)
    elif model == 'erosion':
        image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, dst)
    return image


def connectors(connect_block):
    kernel_size = connect_block.shape[0]
    r = int(kernel_size / 2)

    connect_lines = np.zeros(connect_block.shape)
    for x in range(kernel_size):
        for y in range(kernel_size):
            if connect_block[x, y] != 0:
                l_x = x - r
                l_y = y - r
                f = 0
                h_i, h_j = r, r
                tt = abs(l_x) + abs(l_y)
                connect_line = np.zeros(connect_block.shape)
                while tt:
                    if f == 0:
                        f = 1
                        if l_y:
                            if l_y > 0:
                                h_j += 1
                                l_y -= 1
                                connect_line[h_i, h_j] = 1
                                tt -= 1
                                continue
                            if l_y < 0:
                                h_j -= 1
                                l_y += 1
                                connect_line[h_i, h_j] = 1
                                tt -= 1
                                continue
                    if f == 1:
                        f = 0
                        if l_x:
                            if l_x > 0:
                                h_i += 1
                                l_x -= 1
                                connect_line[h_i, h_j] = 1
                                tt -= 1
                                continue
                            if l_x < 0:
                                h_i -= 1
                                l_x += 1
                                connect_line[h_i, h_j] = 1
                                tt -= 1
                                continue
                connect_lines += connect_line
    return connect_line


def compute_conv(fm, dots, kernel_num=5, l=30):
    kernel = kernel_angel(kernel_num, l=l)
    [h, w] = fm.shape
    r = int(kernel_num / 2)  # 向下取整

    # 添加宽度为1的边缘padding_fm
    padding_fm = np.zeros([h + 2 * r, w + 2 * r])
    rs = np.zeros([h + 2 * r, w + 2 * r])
    padding_fm[r:h + r, r:w + r] = fm

    for dot in tqdm(dots):
        x_0 = dot[0]
        x_1 = dot[0] + 2 * r + 1
        y_0 = dot[1]
        y_1 = dot[1] + 2 * r + 1
        connect_line = connectors(padding_fm[x_0:x_1, y_0:y_1] * kernel)
        rs[x_0:x_1, y_0:y_1] += connect_line

    padding_fm = rs + padding_fm
    padding_fm[padding_fm > 0] = 1
    results = padding_fm[r:h + r, r:w + r]
    rs = np.int8(rs) * 255
    cv2.imwrite(r'C:\Users\pc\Desktop\img_test\org\387_test.tif', rs)
    return results


def road_connect(image, kernel_num=5, l=30):
    """
    道路连接
    :param image: 输入图片
    :param kernel_num: 最大连接距离 kernel_num为奇数
    :param l: 最大为l = kernel_num/2.41
    :return:
    """
    image_np = np.asarray(image, dtype=int)
    h, w = image_np.shape
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dots = []
    for i in tqdm(contours):
        for j in i.reshape(-1, 2):
            tr = j[1]
            j[1] = j[0]
            j[0] = tr
            if 0 in j:
                continue
            if j[0] == w - 1:
                continue
            if j[1] == h - 1:
                continue
            dots.append(list(j))

    image_one = np.int8(image_np / 255)
    connect_one = compute_conv(image_one, dots, kernel_num=kernel_num, l=l)
    connect_one = np.int8(connect_one) * 255
    return connect_one


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
    out.astype('uint8')
    if mode == 'fill':
        ori_img = ori_img + out
    elif mode == 'drop':
        ori_img = ori_img - out
    return ori_img




if __name__ == '__main__':
    """
    道路栅格后处理
    """

    image_path = r'/data/chenyuxia/outputs_09_07/res.tif'
    out_path1 = r'/data/chenyuxia/outputs_09_07/road_mask.tif'
    out_path2 = r'/data/chenyuxia/outputs_09_07/road_mask2.tif'
    out_path3 = r'/data/chenyuxia/outputs_09_07/road_mask3.tif'
    img = cv2.imread(image_path, 0)
    print(img.shape)
    # img = (255 - img)img = img.astype('uint8')
    img = img.astype('uint8')
    imge = fill_drop(img, 100, mode='drop')
    cv2.imwrite(out_path1, imge)
    img = cv2.imread(img, 0)
    img = img.astype('uint8')
    imge = fill_drop(img, 200, mode='fill')
    cv2.imwrite(out_path2, imge)
    img = cv2.imread(img, 0)
    img = img.astype('uint8')
    imge = dilation_erosion(img, kernel_size=5, dst=5, model='dilation')
    # imge = road_connect(img, 31, l=16)  # 道路连接 k为奇数 l = k/2.41
    cv2.imwrite(out_path3, imge)
    # img = cv2.imread(out_path, 0)
    # imge = fill_drop(img, 5000, mode='fill')
    # cv2.imwrite(out_path, imge)
    # img = cv2.imread(out_path, 0)
    # imge = fill_drop(img, 10000, mode='drop')
    # cv2.imwrite(out_path, imge)
