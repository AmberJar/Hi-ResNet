# coding=utf-8
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
import time
import os
import gc
from glob import glob



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

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.409974, 0.433444, 0.446292], [0.187452, 0.144408, 0.161365])

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
                    tmp_img = normalize(to_tensor(tmp_img_0)).unsqueeze(0)
                elif rot==1:
                    tmp_img = normalize(to_tensor(np.rot90(tmp_img_0, 1)).copy()).unsqueeze(0)
                elif rot == 2:
                    tmp_img = normalize(to_tensor(np.rot90(tmp_img_0, 2)).copy()).unsqueeze(0)
                elif rot == 3:
                    tmp_img = normalize(to_tensor(np.rot90(tmp_img_0, 3)).copy()).unsqueeze(0)

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

    # mask.shape = w, h, c
    mask_steps = mask_steps/len(CUTS_PNT)
    mask_steps = np.transpose(mask_steps, (1, 0, 2))

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


def main(config):
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])

    num_classes = 2
    # Model
    print("Load model ............")
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    print("device", device)
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    print(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel

    use_multi_gpu = True
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

    files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))


    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for path in files:
        print(path)
        # cv2 h, w, c
        # outs.shape = h, w, c
        outs = predicts(path, model, output_path=args.output, window_size=1024, rot=0, num_classes=num_classes)
        outs = F.softmax(torch.from_numpy(outs), dim=-1).argmax(-1).cpu().numpy()
        outs = outs.astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, path.split('/')[-1]), outs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_hrnet_road.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNET_HDLOSS/road_second_0.3/best_model.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/inference/taian_data/taian_splited_new_6png', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default=r'/data/fpc/output/outputs_12_14_roads_01/', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-n', '--num_classes', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    _config = json.load(open(args.config))
    main(_config)