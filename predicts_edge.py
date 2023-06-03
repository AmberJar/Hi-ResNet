# coding=utf-8
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import numpy as np
import torch
from torchvision import transforms
import dataloaders
import argparse
import json
import models
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from collections import OrderedDict
import time
import torch.nn.functional as F
from tqdm import tqdm
tic = time.time()
window_size = 512

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# /data/chenyuxia/outputs/river_plant_pg/512/roads/img
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_loveda.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNet_road/05-18_20-29/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/inference/songshanhu/diwu_3/raw', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default=r'/data/fpc/output/outputs_2023_05_19', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='tif', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


args = parse_arguments()
config = json.load(open(args.config))

to_tensor = transforms.ToTensor()
# loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
# normalize = transforms.Normalize([0.402569, 0.40654, 0.401525], [0.135745, 0.140628, 0.127752])
# num_classes = 2
normalize = transforms.Normalize([0.442722, 0.378454, 0.43694], [0.183933, 0.163688, 0.16578])
num_classes = 2

# Model
# print("Load model ............")
# availble_gpus = list(range(torch.cuda.device_count()))
# device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
# print("[device]", device)
# model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
# # Load checkpoint
# checkpoint = torch.load(args.model, map_location=device)
# if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
#     checkpoint = checkpoint['state_dict']
#
# # If during training, we used data parallel
# if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
#     # for gpu inference, use data parallel
#     if "cuda" in device.type:
#         print("HERE")
#         # model = torch.nn.DataParallel(model)
#     # else:
#         # for cpu inference, remove module
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint.items():
#             name = k[7:]
#             new_state_dict[name] = v
#         checkpoint = new_state_dict
# # load
# model.load_state_dict(checkpoint, strict=False)
# model.to(device)
# model.eval() # freeze BN layer and backward memory
#
# print("Load model complete.>>>")
# x = torch.rand(1, 3, 512, 512).cuda()  # dummy data
# traced_script_module = torch.jit.trace(model, x)
# traced_script_module.save(r"/data/chenyuxia/plant_model.pt")
# print("torch.jit save model complete.>>>")

# print("Load model ... >>>")
# availble_gpus = list(range(torch.cuda.device_count()))
# device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
# model = torch.jit.load(r"/data/chenyuxia/weights/river.pt")
# model.to(device)
# model.eval()
# print("Load model complete.>>>")


# Model
print("Load model ............")
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
# model = get_instance(models, 'arch', config, num_classes)
# print(model)
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda' if len(availble_gpus) > 0 else 'cpu')
# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']
# print(checkpoint)
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
    CUTS_PNT = [x for x in range(0, window_size+1, 64)]
    for cuts in CUTS_PNT:
        print((cuts, window_size - w % window_size - cuts), (cuts, window_size - h % window_size - cuts))

        images_copy = np.pad(images, ((cuts, adjust_pad(window_size - w % window_size - cuts)),
                                      (cuts, adjust_pad(window_size - h % window_size - cuts)), (0, 0)),
                             mode="constant")
        nw, nh, c = images_copy.shape
        print(nw, nh, c)

        # 分割 + 重建
        slices_outs = []
        for st_w in range(0, nw, window_size):
            print(st_w, nw)
            batch_data = []
            for st_h in range(0, nh, window_size):
                tmp_img = images_copy[st_w:st_w + window_size, st_h:st_h + window_size, :]
                tmp_img = normalize(to_tensor(tmp_img)).unsqueeze(0)  # to_tensor(tmp_img).unsqueeze(0)
                batch_data.append(tmp_img)

            inputs = torch.cat(batch_data, dim=0)
            with torch.no_grad():
                preds, _ = model(inputs.cuda())
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


from glob import glob

files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
tbar = tqdm(files, ncols=100)
all_iou = []
all_acc = []

if not os.path.exists(args.output):
    os.mkdir(args.output)

for path in tbar:
    save_path = os.path.join(args.output, path.split('/')[-1].split('.')[0] + '.png')
    save_path_1 = os.path.join(args.output, path.split('/')[-1].split('.')[0] + '_p.png')
    label = predicts(path, model)
    # img = cv2.imread(path, 1)
    # pd = np.zeros(img.shape)
    label = F.softmax(torch.from_numpy(label), dim=-1)
    print(label.shape)
    label = label.detach().cpu().numpy()
    label = label[..., 1]
    label = label * 255
    cv2.imwrite(save_path_1, label.astype(np.uint8))
    # label = label.argmax(-1).cpu().numpy()
    label[label < 160] = 0
    label[label >= 160] = 255

    # print(save_path)
    # /data/chenyuxia/outputs/river_plant_pg/512/roads/result
    cv2.imwrite(save_path, label.astype(np.uint8))
# toc = time.time()
# print(toc - tic)