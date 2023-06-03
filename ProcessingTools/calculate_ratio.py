import cv2
from collections import Counter
import os
import numpy as np
from PIL import Image
import time
import shutil
from tqdm import tqdm


label_path = '/data/fpc/data/balance_11classes/labels'

x = [0] * 11
for i in tqdm(os.listdir(label_path)):
    img = cv2.imread(os.path.join(label_path, i), 0)
    img = img.flatten()

    for j in range(11):
        x[j] += Counter(img)[j]

print(x)
