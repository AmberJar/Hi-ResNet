# import cv2
# from collections import Counter
# import os
# import numpy as np
# from PIL import Image
# import time
# import shutil
#
# path = '/data/chenyuxia/new/D14mask_filtered'
# k = 0
# q = 0
# for i in os.listdir(path):
#     img = np.asarray(cv2.imread(os.path.join(path, i), -1), dtype=np.int8).flatten()
#     if Counter(img)[4] > 0:
#         k += 1
#         print(k)
#     q += 1
#     if q % 1000 == 0:
#         print('cur_img_num: ', q)
#
# print(k)
###############
# 根据面积筛选
##############
import cv2
from collections import Counter
import os
import numpy as np
from PIL import Image
import time
import shutil


def label_filter(images_path, labels_path, img_save_path, label_save_path):
    start = time.time()
    k = 0
    for i in os.listdir(images_path):
        j = i[:-3] + 'png'
        image_path = os.path.join(images_path, i)
        label_path = os.path.join(labels_path, j)

        #         print(image_path)
        #         print(label_path)

        image_save = os.path.join(img_save_path, i)
        label_save = os.path.join(label_save_path, j)

        if not os.path.exists(img_save_path):  # 如果路径不存在
            os.makedirs(img_save_path)
        if not os.path.exists(label_save_path):  # 如果路径不存在
            os.makedirs(label_save_path)

        # img = cv2.imread(os.path.join(images_path, i), -1)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(gray)
        print(cv2.imread(label_path).shape)
        img = np.asarray(cv2.imread(label_path), dtype=np.int8).flatten()
        #         print(Counter(img))
        #         print(Counter(img)[3] / img.shape[0])
        ratio = Counter(img)[0] / img.shape[0]

        if ratio < 0.9:
            shutil.copy(image_path, image_save)
            shutil.copy(label_path, label_save)

        if k % 1000 == 0:
            print(time.time() - start)
        k += 1

    print(k)


if __name__ == "__main__":
    images_path = '/data/chenyuxia/roads/deeplobe_filter/all_roads/images_text'
    labels_path = '/data/chenyuxia/roads/deeplobe_filter/all_roads/labels_text'
    img_save_path = '/data/chenyuxia/roads/deeplobe_filter/all_roads/images_filteres'
    label_save_path = '/data/chenyuxia/roads/deeplobe_filter/all_roads/labels_filteres'

    label_filter(images_path, labels_path, img_save_path, label_save_path)



