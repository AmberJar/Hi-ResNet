import os
import shutil
from tqdm import tqdm

# 分离同一个文件夹下的 images 和 labels
def split_train_label(path):
    path_list = os.listdir(path)
    for i in path_list:
        pic_path = os.path.join('/data/fpc/data/deepGlobe/images')
        ana_path = os.path.join('/data/fpc/data/deepGlobe/labels')
        # 后半部分
        behind = i.split('_')[1]
        print(behind)
        if behind == 'sat.jpg':
            shutil.copy(os.path.join(path, i), os.path.join(pic_path, i))
        elif behind == 'mask.png':
            shutil.copy(os.path.join(path, i), os.path.join(ana_path, i))


def change_name(path, save_path):
    for i in tqdm(os.listdir(path)):
        name = i.split('.')[0][:-5] + '.png'
        # print(name)
        # print(os.path.join(save_path, name))
        shutil.copy(os.path.join(path, i), os.path.join(save_path, name))


if __name__ == "__main__":
    # split_train_label('/data/fpc/data/deepGlobe/raw')
    change_name(r'/data/fpc/data/deepGlobe/labels', r'/data/fpc/data/deepGlobe/labels_')