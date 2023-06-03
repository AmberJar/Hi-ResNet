# coding=utf-8
import sys

import ttach as tta
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import numpy as np
import torch
from torchvision import transforms
import dataloaders
import argparse
import json
import models

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from collections import OrderedDict
import time
import torch.nn.functional as F
from tqdm import tqdm


pos = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
test = np.zeros(pos.shape)
pos = np.logical_or(pos, test)  # 或
neg = np.logical_not(pos)  # 非

print(pos)
print(neg)