import numpy as np
import cv2
import os
import time


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
        cv2.imwrite(os.path.join(image_save_file, 'road_images_{}.png'.format(i)), pic_res)
        cv2.imwrite(os.path.join(mask_save_file, 'road_images_{}.png'.format(i)), mask_res)

        # 删除选择过的index
        img_list.remove(pics[0])
        img_list.remove(pics[1])
        img_list.remove(pics[2])
        img_list.remove(pics[3])

    print(time.time() - start)


if __name__ == '__main__':
    images_path = '/data/fpc/data/deep_512/images'
    mask_path = '/data/fpc/data/deep_512/labels'
    img_save_path = '/data/fpc/data/deep_512/images_512'
    ana_save_path = '/data/fpc/data/deep_512/labels_512'
    random_concat(images_path, mask_path, img_save_path, ana_save_path)