import numpy as np
import random
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
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
    sum_ = b + g + r
    # for i in range(m):
    #     for j in range(n):
    #         sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
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

    sum_b, sum_g, sum_r = b, g, r

    sum_b[sum_b < key] = 0
    sum_g[sum_g < key] = 0
    sum_r[sum_r < key] = 0

    # for i in range(m):
    #     for j in range(n):
    #         if sum_[i][j] >= key:
    #             sum_b += b[i][j]
    #             sum_g += g[i][j]
    #             sum_r += r[i][j]
    #             time = time + 1

    avg_b = np.mean(sum_b)
    avg_g = np.mean(sum_g)
    avg_r = np.mean(sum_r)

    maxvalue = float(np.max(img))
    # maxvalue = 255

    b = b * maxvalue / int(avg_b)
    g = g * maxvalue / int(avg_g)
    r = r * maxvalue / int(avg_r)
    b[b > 255] = 255
    b[b < 0] = 0
    g[g > 255] = 255
    g[g < 0] = 0
    r[r > 255] = 255
    r[r < 0] = 0

    # for i in range(m):
    #     for j in range(n):
    #         b = int(img[i][j][0]) * maxvalue / int(avg_b)
    #         g = int(img[i][j][1]) * maxvalue / int(avg_g)
    #         r = int(img[i][j][2]) * maxvalue / int(avg_r)
    #         if b > 255:
    #             b = 255
    #         if b < 0:
    #             b = 0
    #         if g > 255:
    #             g = 255
    #         if g < 0:
    #             g = 0
    #         if r > 255:
    #             r = 255
    #         if r < 0:
    #             r = 0
    #         img[i][j][0] = b
    #         img[i][j][1] = g
    #         img[i][j][2] = r

    return cv2.merge([b, g, r])


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

    Ba[Ba > 255] = 255
    Ga[Ga > 255] = 255
    Ra[Ra > 255] = 255
    # for i in range(len(Ba)):
    #     for j in range(len(Ba[0])):
    #         Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
    #         Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
    #         Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


def white_balance_4(img):
    '''
    基于图像分析的偏色检测及颜色校正方法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''

    b, g, r = cv2.split(img)
    I_r_2 = r ** 2
    I_b_2 = b ** 2
    sum_I_r_2 = np.sum(I_r_2)
    sum_I_b_2 = np.sum(I_b_2)

    sum_I_g = np.sum(g)
    sum_I_r = np.sum(r)
    sum_I_b = np.sum(b)

    max_I_r = np.ceil(np.max(r))
    max_I_r_2 = np.ceil(np.max(I_r_2))
    max_I_g = np.ceil(np.max(g))
    max_I_b_2 = np.max(I_b_2)
    max_I_b = np.ceil(np.max(b))

    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])

    b = u_b * (b**2) + v_b * b
    b[b > 255] = 255
    b[b < 0] = 0
    b = b.astype('int8')

    r = u_r * (r**2) + v_r * r
    r[r > 255] = 255
    r[r < 0] = 0
    r = r.astype('int8')

    g = g.astype('int8')

    return cv2.merge([b, g, r])


def white_balance_5(img):
    '''
    动态阈值算法
    算法分为两个步骤：白点检测和白点调整。
    只是白点检测不是与完美反射算法相同的认为最亮的点为白点，而是通过另外的规则确定
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''

    b, g, r = cv2.split(img)
    """
    YUV空间
    """

    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0

    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]

    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)

    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # 如果该值过大过小，色温向两极端发展
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0

            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.1 * ysum:  # 取前10%的亮点为计算值，如果该值过大易过曝光，该值过小调整幅度小
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb

    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            if b_point > 255:
                b[i][j] = 255
            else:
                if b_point < 0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point > 255:
                g[i][j] = 255
            else:
                if g_point < 0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point > 255:
                r[i][j] = 255
            else:
                if r_point < 0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point

    return cv2.merge([b, g, r])


'''
img : 原图
img1：均值白平衡法
img2: 完美反射
img3: 灰度世界假设
img4: 基于图像分析的偏色检测及颜色校正方法
img5: 动态阈值算法
'''
# img = cv2.imread('./dataset/1/3.JPG')
# # img = cv2.imread('./dataset/2/1_'+str(i)+'.JPG')
# img1 = white_balance_1(img)
# img2 = white_balance_2(img)
# img3 = white_balance_3(img)
# img4 = white_balance_4(img)
# img5 = white_balance_5(img)

img_path = '/data/fpc/inference/taibei'
save_path = '/data/fpc/inference/'


def while_prefer(img_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    k = 0
    import time
    start = time.time()
    for img in os.listdir(img_path):
        pic = cv2.imread(os.path.join(img_path, img))
        ana_path = os.path.join(save_path, img)
        img_ = white_balance_3(pic)
        cv2.imwrite(ana_path, img_)
        if k % 1000 == 0:
            print(time.time() - start)
        k += 1


if __name__ == "__main__":
    while_prefer(img_path, save_path)