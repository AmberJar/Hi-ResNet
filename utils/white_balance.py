import numpy as np
import cv2
import os
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

def while_prefer(img_path, save_path):
    k = 0
    import time
    start = time.time()
    for img in os.listdir(img_path):
        pic = cv2.imread(os.path.join(img_path, img), -1)
        ana_path = os.path.join(save_path, img)
        img_ = white_balance_3(pic)
        cv2.imwrite(ana_path, img_)
        print(k)
        print(time.time() - start)
        k += 1

if __name__ == '__main__':
    print(while_prefer('/data/chenyuxia/new/taibei','/data/chenyuxia/tianjin/taibei_white'))