# coding=utf-8
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
    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNet_Loveda/04-24_15-30/checkpoint-epoch45.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/data/love_DA/Test', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    args = parser.parse_args()
    return args



args = parse_arguments()
config = json.load(open(args.config))
to_tensor = transforms.ToTensor()

normalize = transforms.Normalize([0.280082, 0.299398, 0.307035], [0.127366, 0.109451, 0.115518])
num_classes = 7

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
        preds = model(images.cuda())

        # raw_predictions = F.softmax(torch.from_numpy(preds), dim=-1)
        raw_predictions = torch.softmax(preds, dim=1)
        predictions = raw_predictions.argmax(1)

    return predictions


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

from glob import glob

files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
tbar = tqdm(files, ncols=100)

for i, path in enumerate(tbar):
    label = predicts(path, model)
    label = label.squeeze(0).cpu().numpy()
    # label_rgb = pv2rgb(label)
    save_label_path = r'/data/fpc/data/love_DA/test_results/' + path[len(args.images): -len(
         args.extension)] + "png"
    # # # print(save_path)
    # save_labelrgb_path = r'/mnt/data/chenyuxia/EXPERIMENTAL/vaihingen/test/label_rgb/' + path[len(args.images): -len(
    #      args.extension)] + "png"
    # # # /data/chenyuxia/outputs/river_plant_pg/512/roads/result
    cv2.imwrite(save_label_path, label.astype(np.uint8))
    # cv2.imwrite(save_labelrgb_path, label_rgb.astype(np.uint8))

# toc = time.time()
# print(toc - tic)
