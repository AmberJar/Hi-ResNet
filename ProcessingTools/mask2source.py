import os
import sys

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg


def mask2source(image_path, mask_path, output_path, alpha):
    img = cv2.imread(image_path, 1)

    label = cv2.imread(mask_path,0)
    print(np.unique(label))
    pd = np.zeros(img.shape)
    # label[label != 0] = 1

    pd[np.where(label == 0)] = [0, 0, 0]  # 黑色
    pd[np.where(label == 1)] = [0,0,255] # 深蓝 水体
    pd[np.where(label == 2)] = [0,0,0] #
    pd[np.where(label == 3)] = (0,0,0) #
    pd[np.where(label == 4)] = [255,0,0] # 红色 停车场
    pd[np.where(label == 5)] = [255,192,203] # 粉色 操场
    pd[np.where(label == 6)] = [0,255,255] # Cyan 耕地
    pd[np.where(label == 7)] = [255,246,143] # khaki 大棚
    pd[np.where(label == 8)] = [0, 255, 0] # 浅绿 矮植被
    pd[np.where(label == 9)] = [34, 139, 34] # 深绿 高植被
    pd[np.where(label == 10)] = [205, 133, 0] # 橙色3 裸土

    print(pd.shape)
    pd = cv2.cvtColor(pd.astype(np.uint8), cv2.COLOR_RGB2BGR)

    img_merge = alpha * img + (1 - alpha) * pd

    cv2.imwrite(output_path, img_merge.astype(np.uint8))


if __name__ == '__main__':
    mask_pth = '/data/fpc/inference/merged_mask.png'
    img_path = '/data/fpc/inference/taian_data/raw_pic'
    out_path = '/data/fpc/inference/res/plants_mixed_res'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # list = ['110742_sat.png', '107770_sat.png','109942_sat.png','110190_sat.png','114197_sat.png']
    for i in os.listdir(img_path):
        input_image = os.path.join(img_path, i)
        print(input_image)
        j = i.split('.')[0] + '.png'
        # input_mask = os.path.join(mask_pth, j)
        input_mask = '/data/fpc/inference/merged_mask.png'
        output = os.path.join(out_path, '00006_{}'.format(j))

        mask2source(input_image, input_mask, output, 0.6)