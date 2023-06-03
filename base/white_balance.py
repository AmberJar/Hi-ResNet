import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch


def white_balance_1(img):
    # 读取图像
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img


def white_balance_2(img_input):
    '''
    完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r

    return img

def white_balance_3(img):
    '''
    灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)

    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


# def white_balance_4(img):
#     '''
#     基于图像分析的偏色检测及颜色校正方法
#     :param img: cv2.imread读取的图片数据
#     :return: 返回的白平衡结果图片数据
#     '''
#
#     def detection(img):
#         '''计算偏色值'''
#         img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(img_lab)
#         d_a, d_b, M_a, M_b = 0, 0, 0, 0
#         for i in range(m):
#             for j in range(n):
#                 d_a = d_a + a[i][j]
#                 d_b = d_b + b[i][j]
#         d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
#         D = np.sqrt((np.square(d_a) + np.square(d_b)))
#
#         for i in range(m):
#             for j in range(n):
#                 M_a = np.abs(a[i][j] - d_a - 128) + M_a
#                 M_b = np.abs(b[i][j] - d_b - 128) + M_b
#
#         M_a, M_b = M_a / (m * n), M_b / (m * n)
#         M = np.sqrt((np.square(M_a) + np.square(M_b)))
#         k = D / M
#         print('偏色值:%f' % k)
#         return
#
#     b, g, r = cv2.split(img)
#     # print(img.shape)
#     m, n = b.shape
#     # detection(img)
#
#     I_r_2 = np.zeros(r.shape)
#     I_b_2 = np.zeros(b.shape)
#     sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
#     max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(
#         b[0][0]), int(g[0][0])
#     for i in range(m):
#         for j in range(n):
#             I_r_2[i][j] = int(r[i][j] ** 2)
#             I_b_2[i][j] = int(b[i][j] ** 2)
#             sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
#             sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
#             sum_I_g = g[i][j] + sum_I_g
#             sum_I_r = r[i][j] + sum_I_r
#             sum_I_b = b[i][j] + sum_I_b
#             if max_I_r < r[i][j]:
#                 max_I_r = r[i][j]
#             if max_I_r_2 < I_r_2[i][j]:
#                 max_I_r_2 = I_r_2[i][j]
#             if max_I_g < g[i][j]:
#                 max_I_g = g[i][j]
#             if max_I_b_2 < I_b_2[i][j]:
#                 max_I_b_2 = I_b_2[i][j]
#             if max_I_b < b[i][j]:
#                 max_I_b = b[i][j]
#
#     [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
#     [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
#     # print(u_b, v_b, u_r, v_r)
#     b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
#     for i in range(m):
#         for j in range(n):
#             b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
#             g0[i][j] = g[i][j]
#             # r0[i][j] = r[i][j]
#             r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
#             if r_point > 255:
#                 r0[i][j] = 255
#             else:
#                 if r_point < 0:
#                     r0[i][j] = 0
#                 else:
#                     r0[i][j] = r_point
#             if b_point > 255:
#                 b0[i][j] = 255
#             else:
#                 if b_point < 0:
#                     b0[i][j] = 0
#                 else:
#                     b0[i][j] = b_point
#     return cv2.merge([b0, g0, r0])


if __name__ == '__main__':
    image = cv2.imread("huawei-dataset/train/images/0.tif")
    image = white_balance_3(image)
    cv2.imwrite("0_10_img.png", image)