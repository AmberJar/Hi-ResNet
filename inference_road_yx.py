# coding=utf-8
import sys
from glob import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
import dataloaders
import argparse
import json
import models
import os
from collections import OrderedDict
import time
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from hrnet_trainer_f import *
import math
from collections import Counter

tic = time.time()
window_size = 512
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_hrnet_road_yx.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    # old road with conn /data/fpc/saved/HRNET_HDLOSS/road_second_0.3/best_model.pth
    # new road without conn /data/fpc/saved/HRNet_Road/road_raw_deep_3w_res/checkpoint-epoch39.pth
    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNet_Road/road_raw_deep_3w_res/checkpoint-epoch39.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/inference/piggy', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default=r'/data/fpc/output/outputs_12_08_roads_yx/', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


args = parse_arguments()
config = json.load(open(args.config))

to_tensor = transforms.ToTensor()
loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
normalize = transforms.Normalize([0.402568, 0.406539, 0.401525], [0.135746, 0.140629, 0.127753])
num_classes = 2
# old road
# [0.402568, 0.406539, 0.401525]
# [0.135746, 0.140629, 0.127753]
# new road
# [0.442722, 0.378454, 0.43694]
# [0.183933, 0.163688, 0.16578]
'''
# Model
print("Load model ............")
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
print("[device]", device)
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

# If during training, we used data parallel
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    # for gpu inference, use data parallel
    if "cuda" in device.type:
        print("HERE")
        # model = torch.nn.DataParallel(model)
    # else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict
# load
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval() # freeze BN layer and backward memory

print("Load model complete.>>>")
x = torch.rand(1, 3, 512, 512).cuda()  # dummy data
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save(r"/data/fpc/plant_model.pt")
print("torch.jit save model complete.>>>")

sys.exit()
'''

# Model
print("Load model ............")
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:3' if len(availble_gpus) > 0 else 'cpu')
print("device", device)
# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
print(args.model)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']
# If during training, we used data parallel

use_multi_gpu = False
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    # for gpu inference, use data parallel
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
    else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            # print(k)
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict
# load
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("Load model complete.>>>")


def adjust_pad(ele):
    if ele < 0:
        return ele + window_size
    else:
        return ele


def predicts(image_path, model):
    # print(image_path)
    images = cv2.imread(image_path)
    # print(images.shape)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    w, h, c = images.shape
    mask_steps = np.zeros((w, h, num_classes))
    CUTS_PNT = [x for x in range(0, window_size + 1, 64)]
    for cuts in CUTS_PNT:
        # print((cuts, window_size - w % window_size - cuts), (cuts, window_size - h % window_size - cuts))

        images_copy = np.pad(images, ((cuts, adjust_pad(window_size - w % window_size - cuts)),
                                      (cuts, adjust_pad(window_size - h % window_size - cuts)), (0, 0)),
                             mode="constant")
        nw, nh, c = images_copy.shape
        # print(nw, nh, c)

        # 分割 + 重建
        slices_outs = []
        for st_w in range(0, nw, window_size):
            # print(st_w, nw)
            batch_data = []
            for st_h in range(0, nh, window_size):
                tmp_img = images_copy[st_w:st_w + window_size, st_h:st_h + window_size, :]
                tmp_img = normalize(to_tensor(tmp_img)).unsqueeze(0)  # to_tensor(tmp_img).unsqueeze(0)
                batch_data.append(tmp_img)

            inputs = torch.cat(batch_data, dim=0)
            with torch.no_grad():
                inputs = inputs.to(device)
                # print("inputs", inputs.device)
                preds, _ = model(inputs)
                # print('pp:', preds.shape)
                preds = torch.softmax(preds, dim=1).detach().cpu().numpy()

            # print(preds.shape)
            outs = [ele for ele in preds]
            outs = np.concatenate(outs, axis=-1)
            slices_outs.append(outs)
            # print(outs.shape)

        slices_outs = np.concatenate(slices_outs, axis=1)
        # print("slices_outs.shape", slices_outs.shape)
        slices_outs = slices_outs.transpose([1, 2, 0])
        slices_outs = slices_outs[cuts:cuts + w, cuts: cuts + h, :]
        # cv2.imwrite('/data/chenyuxia/road_text' + path[len(args.images): -len(args.extension)] + "tif", (slices_outs[:, :, 1] * 255).astype(np.uint8))
        # print('here')
        # print(slices_outs.shape)
        mask_steps += slices_outs

    mask_steps = mask_steps / len(CUTS_PNT)
    # print('here too')
    # print(mask_steps.shape)
    return mask_steps


def soft_iou_score(y_true, y_pred):
    epsilon = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)


def acc_score(y_true, y_pred):
    epsilon = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return np.equal(y_true_f, y_pred_f).sum() / len(y_true_f)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':
    files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for path in tqdm(files):
        outs = predicts(path, model)
        outs = F.softmax(torch.from_numpy(outs), dim=-1).argmax(-1).cpu().numpy()
        outs = outs.astype(np.uint8)

        # outs[outs <= 50] = 0
        # outs[outs > 50] = 1
        # outs = outs.astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, path.split('/')[-1]), outs)

        # yuxia_path = '/'
        # for item in path.split('/'):
        #     if item == 'img':
        #         item = 'label'
        #     yuxia_path = os.path.join(yuxia_path, item)
        #
        # yuxia_path = yuxia_path.split('.')[0] + '.png'
        # yuxia = cv2.imread(yuxia_path, 0)
        # print(np.unique(yuxia))
        # test_list = [255]
        #
        # for i in test_list:
        #     print(i)
        #     target = yuxia.copy()
        #
        #     target[target != i] = 0
        #     target[target == i] = 1
        #
        #     print(np.unique(target))
        #     pre_im = outs.copy()
        #     pre_im[pre_im != i] = 0
        #     pre_im[pre_im == i] = 1
        #
        #     metrics_function = Evaluator(num_class=2)
        #     metrics_function.confusion_matrix = metrics_function._generate_matrix(target, pre_im)
        #     iou = metrics_function.Mean_Intersection_over_Union()
        #     acc = metrics_function.Pixel_Accuracy_Class()
        #
        #     # iou = soft_iou_score(target, pre_im)
        #     # acc = acc_score(target, pre_im)
        #     print('iou', iou)
        #     print('acc', acc)
        #
        # print('========================================')

