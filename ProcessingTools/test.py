import cv2
import os
from tqdm import tqdm
path = '/data/fpc/data/deepGlobe/deep/trainset/val/labels'
save_path = '/data/fpc/data/deepGlobe/deep/trainset/val/labels_'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in tqdm(os.listdir(path)):
    label = cv2.imread(os.path.join(path, i), 0)
    label[label != 0] = 1

    cv2.imwrite(os.path.join(save_path, i), label)