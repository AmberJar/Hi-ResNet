import cv2
import os
import numpy as np
from PIL import Image
def images_merge(file,part):
    pic_list = []
    for i in range(part**2):
        img_split_nam = 'split_' + str(i) + 'last' + '.tif'
        name = os.path.join(file, img_split_nam)
        img = cv2.imread(name, -1)
        pic_list.append(img)
    # print(pic_list)
    img_y = []
    for i in range(0, part**2, part):
        # print(pic_list[i].shape, pic_list[i+1].shape,pic_list[i+2].shape, pic_list[i+3].shape)
        img_y1 = np.concatenate((pic_list[i], pic_list[i+1]), axis=1)
        img_y2 = np.concatenate((pic_list[i+2], pic_list[i+3]), axis=1)
        img_y.append(np.concatenate((img_y1, img_y2), axis=1))
    img_x = img_y[0]
    for j in range(1, part):
        img_x = np.concatenate((img_x, img_y[j]), axis=0)
    print(img_x.shape)
    cv2.imwrite('/data/chenyuxia/outputs_08_22-1/res.tif', img_x)

    #
    # img_x = np.concatenate((pic_list[0], pic_list[1]), axis=1)  # axis=0 按垂直方向，axis=1 按水平方向
    # img_y = np.concatenate((pic_list[2], pic_list[3]), axis=1)
    # res = np.concatenate((img_x, img_y), axis=0)
    # cv2.imwrite('/data/chenyuxia/outputs_08_11-1/res2.tif', res)
if __name__ == "__main__":
    # name = cv2.imread('/data/chenyuxia/outputs_08_22-1/res2.tif',-1)
    # print(name.shape)
    print(images_merge('/data/chenyuxia/outputs_08_22-1', 4))

