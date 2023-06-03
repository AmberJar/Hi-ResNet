import gc
import os
import time

import numpy as np
import torch
from torchvision import transforms
import dataloaders
import argparse
import json
import models
from collections import OrderedDict
import rasterio
from tqdm import tqdm
import cv2
import torch.nn.functional as F


def adjust_pad(ele, window_size):
    if ele<0:
        return ele + window_size
    else:
        return ele


def predict_on_batchsize(models, inputs, batchsize=64):
    outs = []
    b, c, w, h = inputs.shape
    b_1st = b % batchsize
    cnt_range = b // batchsize
    if b_1st>0:
        with torch.no_grad():
            temp_data = inputs[:b_1st, :, :, :]
            _, preds = models(temp_data.cuda())
            outs.append(preds)

    for iter in range(0, cnt_range):
        temp_data = inputs[b_1st + iter*batchsize:b_1st + (iter+1) * batchsize, :, :, :]
        with torch.no_grad():
            _, pred = models(temp_data.cuda())
            outs.append(preds)
    outs = torch.cat(outs, dim=0)
    return outs


def predicts(image_path, model, output_path, window_size=256, rot=0, batchsize=128, num_classes=2):
    '''
    rot 旋转N个90°
    '''
    assert rot in [0, 1, 2, 3]
    with rasterio.open(image_path) as raster_src:
        data = raster_src.read()
        raster_crs = raster_src.crs
        raster_transform = raster_src.transform

    data = np.transpose(data, (2, 1, 0)) # RGB, --> w,h,c
    w, h, c = data.shape

    mask_steps = np.zeros((w, h, num_classes))
    CUTS_PNT = [x for x in range(0, window_size, 64)]

    for cuts in tqdm(CUTS_PNT, desc="当前图片分割进度", colour='blue', position=1):
        images_copy = np.pad(data, (
            (cuts, adjust_pad(window_size-w%window_size-cuts, window_size)),
            (cuts, adjust_pad(window_size-h%window_size-cuts, window_size)),
            (0, 0)
        ), mode="constant")

        nw, nh, c = images_copy.shape

        # 分割 + 重建
        slices_outs = []
        for st_w in range(0, nw, window_size):
            batch_data = []

            for st_h in range(0, nh, window_size):
                tmp_img_0 = images_copy[st_w:st_w+window_size, st_h:st_h+window_size, :3]

                if rot == 0:
                    tmp_img = to_tensor(tmp_img_0).unsqueeze(0)
                elif rot==1:
                    tmp_img = to_tensor(np.rot90(tmp_img_0, 1).copy()).unsqueeze(0)
                elif rot == 2:
                    tmp_img = to_tensor(np.rot90(tmp_img_0, 2).copy()).unsqueeze(0)
                elif rot == 3:
                    tmp_img = to_tensor(np.rot90(tmp_img_0, 3).copy()).unsqueeze(0)

                batch_data.append(tmp_img)

            inputs = torch.cat(batch_data, dim=0)

            preds = predict_on_batchsize(models=model, inputs=inputs, batchsize=batchsize)
            preds = torch.softmax(preds, dim=1).detach().cpu().numpy()
            # with torch.no_grad():
            #     preds, _ = model(inputs.cuda())
            #     preds = torch.softmax(preds, dim=1).detach().cpu().numpy()

            if rot==0:
                outs = [ele for ele in preds]
            elif rot==1:
                outs = [np.rot90(ele.transpose([1, 2, 0]), 3).transpose([2, 0, 1]) for ele in preds]
            elif rot == 2:
                outs = [np.rot90(ele.transpose([1, 2, 0]), 2).transpose([2, 0, 1]) for ele in preds]
            elif rot == 3:
                outs = [np.rot90(ele.transpose([1, 2, 0]), 1).transpose([2, 0, 1]) for ele in preds]
            outs = np.concatenate(outs, axis=-1)
            slices_outs.append(outs)

        slices_outs = np.concatenate(slices_outs, axis=1)
        slices_outs = slices_outs.transpose([1, 2, 0])
        slices_outs = slices_outs[cuts:cuts+w, cuts: cuts+h, :]
        mask_steps += slices_outs

    del slices_outs, data, images_copy
    gc.collect()

    mask_steps = mask_steps/len(CUTS_PNT)
    mask_steps = np.transpose(mask_steps, (2, 1, 0))

    # im_bands, im_height, im_width = mask_steps.shape
    # dst_img_0 = output_path + "/" + image_path.split("/")[-1][:-len(args.extension)] + "_rot_{}.tif".format(rot)  # 输出路径
    # with rasterio.open(dst_img_0, 'w', driver='GTiff',  # 图像类型
    #     height=im_height, width=im_width,
    #     count=1,  # 总层数
    #     dtype=np.uint8,  # 数据类型
    #     crs=raster_crs,  transform=raster_transform) as dataset:
    #     dataset.write((mask_steps[1, :, :] * 255).astype(np.uint8), 1)

    # return (mask_steps[1, :, :] * 255).astype(np.uint8)
    return mask_steps


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_uppernet.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='/data/fpc/saved/UpperNetA/11-03_23-38/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default='/data/fpc/inference/test/taibei_test', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='/data/fpc/output/outputs_10_24_1', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


# =======================================================================================================
# =======================================================================================================

args = parse_arguments()
config = json.load(open(args.config))

to_tensor = transforms.ToTensor()
loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
num_classes = loader.dataset.num_classes

# Model
print("Load model ............")
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

# ========
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
# ========
# If during training, we used data parallel
# if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
#     # for gpu inference, use data parallel
#     if "cuda" in device.type:
#         model = torch.nn.DataParallel(model)
#     else:
#         # for cpu inference, remove module
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint.items():
#             name = k[7:]
#             new_state_dict[name] = v
#         checkpoint = new_state_dict
# load
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("Load model complete.>>>")

# x = torch.rand(1, 3, 512, 512).cuda()  # dummy data
# traced_script_module = torch.jit.trace(model, x)
# traced_script_module.save(r"/data/fpc/inference/weights/river.pt")

# model_name = 'uppernet_a'
# dummy_input = torch.rand(1, 3, 512, 512).cuda()
# dummy_output = model(dummy_input)
# # print('output_size', dummy_output.size())
# # model_trace = torch.jit.trace(model, dummy_input)
# # model_script = torch.jit.script(model)
# input_names = 'input1'
# output_names = 'output1'
# # 跟踪法与直接 torch.onnx.export(model, ...)等价
# # torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx', input_names=input_names, output_names=output_names)
# # 记录法必须先调用 torch.jit.sciprt
# path = r'test1.onnx'
# torch.onnx.export(model, dummy_input, path, opset_version=11)
#
# print("torch.jit save model complete.>>>")

if not os.path.exists(args.output):
    os.mkdir(args.output)

start = time.time()
# NoTTA

for i in tqdm(os.listdir(args.images), desc="总进度", colour='green', position=0):
    path = os.path.join(args.images, i)
    res = predicts(path, model, output_path=args.output, window_size=1024, rot=0, num_classes=8)
    # res = res[:6, ...]
    print(res.shape)
    # print(res[7])
    res = np.argmax(res, axis=0)
    print(np.unique(res))
    output_path = os.path.join(args.output, i)
    cv2.imwrite(output_path, res.astype(np.uint8))

# pbar = tqdm(enumerate(os.listdir(args.images)), colour='pink')
# for j, file_path in enumerate(os.listdir(args.images)):
#     # pbar.set_description("进度 %d" % char)
#     path = os.path.join(args.images, file_path)
#     for i in range(4):
#         # print('stage:', i)
#         if i == 0:
#             res = predicts(path, model, output_path=args.output, window_size=1024, rot=i, num_classes=9)
#         else:
#             res += predicts(path, model, output_path=args.output, window_size=1024, rot=i, num_classes=9)
#     res = res / 4
#     res[res <= 70] = 0
#     res[res > 70] = 255
#     print(j)
#     cv2.imwrite(os.path.join(args.output, file_path), res.astype(np.uint8))

print(time.time() - start)






