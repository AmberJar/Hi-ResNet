from tqdm import tqdm
import numpy as np
import os
from PIL import Image
# daSpaceMaps_palette: 黑-白-蓝-棕-红-绿 RGB值
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2


# AdaSpaceMaps_palette = np.array([[166, 202, 240],
#                             [128, 128, 0],
#                             [0, 0, 128],
#                             [255, 0, 0],
#                             [0, 128, 0],
#                             [128, 0, 0],
#                             [255, 233, 233],
#                             [160, 160, 164],
#                             [0, 128, 128],
#                             [90, 87, 255],
#                             [255, 255, 0],
#                             [255, 192, 0],
#                             [0, 0, 255],
#                             [255, 0, 192],
#                             [128, 0, 128],
#                             [0, 255, 0],
#                             [0, 255, 255]])
# AdaSpaceMaps_labels = (['airplane',
#                        'bare soil',
#                         'buildings',
#                         'cars',
#                         'chaparral',
#                         'court',
#                         'dock',
#                         'field',
#                         'grass',
#                         'mobile home',
#                         'pavement',
#                         'sand',
#                         'sea',
#                         'ship',
#                         'tanks',
#                         'trees',
#                         'water'])
def labels2gry(img_path, labels_save):
    if not os.path.exists(labels_save):
        os.mkdir(labels_save)
    AdaSpaceMaps_palette = np.array([[0, 0, 0],
                            [200, 0, 0],
                            [250, 0, 150],
                            [200, 150, 150],
                            [250, 150, 150],
                            [0, 200, 0],
                            [150, 250, 0],
                            [150, 200, 150],
                            [200, 0, 200],
                            [150, 0, 250],
                            [150,  150, 250],
                            [250, 200, 0],
                            [200, 200, 0],
                            [0,  0, 200],
                            [0, 150, 200],
                            [0, 200, 250]])
    AdaSpaceMaps_labels = (['industrial land',
                           'bare soil',
                            'buildings',
                            'cars',
                            'chaparral',
                            'court',
                            'dock',
                            'field',
                            'grass',
                            'mobile home',
                            'pavement',
                            'sand',
                            'sea',
                            'ship',
                            'tanks',
                            'trees',
                            'water'])
    for root, dirs, files in os.walk(img_path):
        img_files = [file_name for file_name in files]
        print(img_files)
        for file in img_files:
            file_path = os.path.join(root, file)
            mask = cv2.imread(file_path, -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = mask.astype(int)
            print(mask.shape)
            labels_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
            for ii, label in enumerate(AdaSpaceMaps_palette):
                locations = np.all(mask == label, axis=-1)
                print(locations)
                labels_mask[locations] = ii
    # outputImg = Image.fromarray(outputImg * 255.0)
    # "L"代表将图片转化为灰度图
    # outputImg = outputImg.convert('L')
    # outputImg.save('/Users/86187/Desktop/change_detection/YR-A-result.bmp')
    # outputImg.show()
            labels_mask.astype(np.uint8)
            cv2.imwrite(os.path.join(labels_save, file), labels_mask)
    return

def label_divice(output_path,classes):
    # labels_mask = labels2gry(r'/data/chenyuxia/tianjin/')
    labels_mask = cv2.imread(r'/data/chenyuxia/taibei/r2.tif',-1) #打开图片
    labels_mask = np.asarray(labels_mask)      #转换为矩阵
    print(labels_mask.shape)
    # image_array是归一化的二维浮点数矩阵
    k = 0
    labels_mask[labels_mask != classes] = 0
    labels_mask[labels_mask == classes] = 255
    if 255 in labels_mask:
        print('255 is here')
    labels_mask.astype(np.uint8)
    cv2.imwrite(r'/data/chenyuxia/road_text/bare_land.tif', labels_mask)
    # im.save(output_path + 'res5' + '.tif')


    # 标签保存
    # cv2.imwrite(gray_save_path + label_name, label_mask)


if __name__ == '__main__':
    # img = cv2.imread(r'/data/chenyuxia/tianjin/res4.tif', -1)
    # print(img.shape)
    # print(labels2gry(r'/data/chenyuxia/road_text/'))
    print(labels2gry(r'/data/chenyuxia/GID/label_15classes/', r'/data/chenyuxia/GID/labels_'))