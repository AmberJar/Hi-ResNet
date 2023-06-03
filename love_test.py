# coding=utf-8
import sys

import ttach as tta
import os
from torchstat import stat
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

tic = time.time()
window_size = 1024
from torch import nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# /data/chenyuxia/outputs/river_plant_pg/512/roads/img
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_loveda.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNet_road/05-18_20-29/best_model.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fangpengcheng/data/vaihingen/test_samples', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-e', '--extension', default='tif', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default="d4", choices=[None, "d4", "lr"])
    args = parser.parse_args()
    return args



args = parse_arguments()
print(args.config)
config = json.load(open(args.config))
to_tensor = transforms.ToTensor()


# loveda [0.463633, 0.316652, 0.320528], [0.203334, 0.135546, 0.140651]
# NAIC [0.29446, 0.300793, 0.315524], [0.115252, 0.095638, 0.1026]
normalize = transforms.Normalize([0.29446, 0.300793, 0.315524], [0.115252, 0.095638, 0.1026])
num_classes = 11

# Model
print("Load model ............")
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
# model = get_instance(models, 'arch', config, num_classes)
# print(model)
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda' if len(availble_gpus) > 0 else 'cpu')
# Load checkpoint
checkpoint = torch.load(args.model)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

# If during training, we used data parallel
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    # for gpu inference, use data parallel
    if "cuda" in device.type:
        model = torch.nn.DataParallel(model)
    else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict

# load

model.to(device)
model.load_state_dict(checkpoint)
model.eval()
print("Load model complete.>>>")


def predicts(image_path, model):
    # print(image_path)
    results = []
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    images = cv2.imread(image_path)
    # print(images.shape)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    images = normalize(to_tensor(images)).unsqueeze(0)
    with torch.no_grad():
        out_aux, out = model(images.cuda())

        # raw_predictions = F.softmax(torch.from_numpy(preds), dim=-1)
        out = torch.softmax(out, dim=1)
        out = out.argmax(1)

        out_aux = torch.softmax(out_aux, dim=1)
        out_aux = out_aux.argmax(1)

    return out_aux, out


def gray2grb(label):
    pd = np.zeros((label.shape[0], label.shape[1], 3))
    print(np.unique(label))
    # label[label != 2] = 0
    pd[np.where(label == 0)] = [197, 235, 204] # 树
    pd[np.where(label == 2)] = [189, 216, 229]  # 道路
    pd[np.where(label == 9)] = [227, 205, 179]  # 深蓝 水体
    pd[np.where(label == 10)] = [242, 242, 242]  # 边边
    # pd[np.where(label == 2)] = [0,0,0] #
    # pd[np.where(label == 3)] = (0,0,0) #
    # pd[np.where(label == 4)] = [255,0,0] # 红色 停车场
    # pd[np.where(label == 5)] = [255,192,203] # 粉色 操场
    # pd[np.where(label == 6)] = [0,255,255] # Cyan 耕地
    # pd[np.where(label == 7)] = [255,246,143] # khaki 大棚
    # pd[np.where(label == 8)] = [0, 255, 0] # 浅绿 矮植被
    # pd[np.where(label == 9)] = [34, 139, 34] # 深绿 高植被
    # pd[np.where(label == 10)] = [205, 133, 0] # 橙色3 裸土

    return pd

# out_aux_, out_ = predicts('/data/fangpengcheng/data/1245.tif', model)
# out_aux_ = out_aux_.squeeze(0).cpu().numpy()
# out_ = out_.squeeze(0).cpu().numpy()
#
# out_ = gray2grb(out_)
# out_aux_ = gray2grb(out_aux_)
#
# cv2.imwrite('/data/fangpengcheng/data/out.png', out_)
# cv2.imwrite('/data/fangpengcheng/data/out_aux.png', out_aux_)
# sys.exit()

out_aux_, out_ = predicts('/data/fangpengcheng/data/1245.tif', model)
out_aux_ = out_aux_.squeeze(0).cpu().numpy()
out_ = out_.squeeze(0).cpu().numpy()

# sys.exit()

from glob import glob

files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
tbar = tqdm(files, ncols=100)

for i, path in enumerate(tbar):
    label = predicts(path, model, i)
    label = label.squeeze(0).cpu().numpy()
    # label_rgb = pv2rgb(label)
    save_label_path = r'/data/fangpengcheng/data/loveDA/test_results/' + path[len(args.images): -len(
         args.extension)] + "png"
    # # # print(save_path)
    # save_labelrgb_path = r'/mnt/data/chenyuxia/EXPERIMENTAL/vaihingen/test/label_rgb/' + path[len(args.images): -len(
    #      args.extension)] + "png"
    # # # /data/chenyuxia/outputs/river_plant_pg/512/roads/result
    cv2.imwrite(save_label_path, label.astype(np.uint8))
    # cv2.imwrite(save_labelrgb_path, label_rgb.astype(np.uint8))

# toc = time.time()
# print(toc - tic)
