import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(pic_path, label_path, train_ratio):
    print(os.getcwd())  # 输出：F:\test\test1\test2
    print(os.path.dirname(pic_path))  # 获取当前文件的父目录，此方法生成的并不是绝对路径，输出：F:/test/test1/test2
    print(os.path.abspath(os.path.dirname(pic_path)))  # 使用os.path.abspath()方法，输出：F:\test\test1\test2

    root = os.path.join(os.path.abspath(os.path.dirname(pic_path)), 'trainset')
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')

    if not os.path.exists(train_path):  # 如果路径不存在
        os.makedirs(train_path)
    if not os.path.exists(val_path):  # 如果路径不存在
        os.makedirs(val_path)

    train_images = os.path.join(train_path, 'images')
    train_labels = os.path.join(train_path, 'labels')
    val_images = os.path.join(val_path, 'images')
    val_labels = os.path.join(val_path, 'labels')

    if not os.path.exists(train_images):  # 如果路径不存在
        os.makedirs(train_images)
    if not os.path.exists(train_labels):  # 如果路径不存在
        os.makedirs(train_labels)
    if not os.path.exists(val_images):  # 如果路径不存在
        os.makedirs(val_images)
    if not os.path.exists(val_labels):  # 如果路径不存在
        os.makedirs(val_labels)

    # 随机数的种子
    seed = 2000

    pathDir = os.listdir(pic_path)  # 取图片的原始路径
    labelDir = os.listdir(label_path)  # 取label的原始路径
    filenumber = len(pathDir)
    # rate = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate = train_ratio

    x_train, x_test = train_test_split(pathDir, train_size=train_ratio, random_state=seed, shuffle=True)

    for name in tqdm(x_train):
        label_name = name.split('.')[0] + '.png'
        shutil.copy(os.path.join(pic_path, name), os.path.join(train_images, name))
        shutil.copy(os.path.join(label_path, label_name), os.path.join(train_labels, label_name))

    for name in tqdm(x_test):
        label_name = name.split('.')[0] + '.png'
        shutil.copy(os.path.join(pic_path, name), os.path.join(val_images, name))
        shutil.copy(os.path.join(label_path, label_name), os.path.join(val_labels, label_name))


split_dataset('/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376_20w/images', '/data/fpc/data/Mapillaryv1_2/mp_pretrain_dataset/trainset_376_20w/labels', 0.9)