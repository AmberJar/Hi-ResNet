import numpy as np
import cv2
import os
import time
import math
def split_image(path):
    for i in os.listdir(path):
        pic_path = os.path.join(path, i)
        img = cv2.imread(pic_path)
        # img = Image.open(pic_path)
        print(img.shape)
        (height, width, _) = img.shape
        height_list = [math.ceil(height / 2), height - math.ceil(height / 2) + 1]
        width_list = [math.ceil(width / 2), width - math.ceil(width / 2) + 1]

        cur_h, pre_h = 0, 0
        k = 0
        for x in range(len(height_list)):
            h = height_list[x]
            pre_h = cur_h
            cur_h += h
            cur_w, pre_w = 0, 0
            for j in range(len(width_list)):
                w = width_list[j]
                pre_w = cur_w
                cur_w += w
                print(pre_h, cur_h - 1, pre_w, cur_w - 1)
                img_ = img[pre_h:cur_h - 1, pre_w:cur_w - 1]
                name = i.split('.')[0]
                cv2.imwrite('/data/chenyuxia/NAIC/labels_/NAIC/airplanelabels_splited/{}_splited_{}.png'.format(name, k), img_)
                k += 1


def random_concat(image, mask, image_save_file, mask_save_file):
    start = time.time()
    if not os.path.exists(image_save_file):
        os.mkdir(image_save_file)
    if not os.path.exists(mask_save_file):
        os.mkdir(mask_save_file)
    img_list = os.listdir(image)
    img_list_length = len(img_list)
    print(img_list_length)
    times = int(img_list_length / 4)
    print(times)
    times = times * 4
    names = locals()
    for i in range(times):
        pics = np.random.choice(img_list, 4, replace=False)

        # 随机读取pic
        pic0 = cv2.imread(os.path.join(image, pics[0]))
        pic1 = cv2.imread(os.path.join(image, pics[1]))
        pic2 = cv2.imread(os.path.join(image, pics[2]))
        pic3 = cv2.imread(os.path.join(image, pics[3]))

        # 随机读取mask
        mask0 = cv2.imread(os.path.join(mask, pics[0]))
        mask1 = cv2.imread(os.path.join(mask, pics[1]))
        mask2 = cv2.imread(os.path.join(mask, pics[2]))
        mask3 = cv2.imread(os.path.join(mask, pics[3]))

        # 合并pic
        pic_tmp1 = np.concatenate((pic0, pic1), axis=0)
        pic_tmp2 = np.concatenate((pic2, pic3), axis=0)
        pic_res = np.concatenate((pic_tmp1, pic_tmp2), axis=1)

        #合并mask
        mask_tmp1 = np.concatenate((pic0, pic1), axis=0)
        mask_tmp2 = np.concatenate((pic2, pic3), axis=0)
        mask_res = np.concatenate((mask_tmp1, mask_tmp2), axis=1)

        # 储存
        cv2.imwrite(os.path.join(image_save_file, 'airplane_images_{}.png'.format(i)), pic_res)
        cv2.imwrite(os.path.join(mask_save_file, 'airplane_images_{}.png'.format(i)), mask_res)

        # # 删除选择过的index
        # img_list.remove(pics[0])
        # img_list.remove(pics[1])
        # img_list.remove(pics[2])
        # img_list.remove(pics[3])

    print(time.time() - start)


if __name__ == '__main__':
    split_image(r'/data/chenyuxia/NAIC/labels_/NAIC/airplane_label')

    # images_path = '/data/chenyuxia/NAIC/labels_/NAIC/airplaneimgs_splited'
    # mask_path = '/data/chenyuxia/NAIC/labels_/NAIC/airplanelabels_splited'
    # img_save_path = '/data/chenyuxia/NAIC/labels_/NAIC/airplaneimg_enhance'
    # ana_save_path = '/data/chenyuxia/NAIC/labels_/NAIC/airplanelabel_enhance'
    # random_concat(images_path, mask_path, img_save_path, ana_save_path)