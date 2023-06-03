import cv2
import os
import numpy as np
from tqdm import tqdm


def img_make_up(img_path, ana_path, img_save_file, ana_save_file):
    for name in tqdm(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, name))
        ana = cv2.imread(os.path.join(ana_path, name), 0)

        h, w, c = img.shape

        if h > 512:
            img = img[0:512, ...]
            ana = ana[0:512, :]
        if w > 512:
            img = img[:, 0:512, :]
            ana = ana[:, 0:512]

        if h < 512 and w < 512:
            img = np.pad(img, ((0, 512 - h), (0, 512 - w), (0, 0)), 'constant')
            ana = np.pad(ana, ((0, 512 - h), (0, 512 - w)), 'constant')
        elif h > w:
            img = np.pad(img,((0, 0), (0, h - w),(0, 0)),'constant')
            ana = np.pad(ana, ((0, 0), (0, h - w)), 'constant')
        else:
            img = np.pad(img, ((0, w - h), (0, 0), (0, 0)), 'constant')
            ana = np.pad(ana, ((0, w - h), (0, 0)), 'constant')

        cv2.imwrite(os.path.join(img_save_file, name), img)
        cv2.imwrite(os.path.join(ana_save_file, name), ana)




# a = np.ones((3,4,8))
# print(a.shape)
# # 第一二三四，分别是，上下，左右
# b = np.pad(a,((0, 0), (0, a.shape[2] - a.shape[1]),(0, 0)),'constant')
# print(b.shape)

if __name__ == '__main__':
    img_make_up('/data/fpc/data/Lung_Carcinoma_Seg/sagittal/images_filtered', '/data/fpc/data/Lung_Carcinoma_Seg/sagittal/labels_filtered',
                '/data/fpc/data/Lung_Carcinoma_Seg/sagittal/images_', '/data/fpc/data/Lung_Carcinoma_Seg/sagittal/labels_')