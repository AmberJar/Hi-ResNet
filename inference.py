import argparse
import sys

import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict
from PIL import ImageFile
import rasterio
import cv2

Image.MAX_IMAGE_PIXELS = None  #读取图像的上限调整
ImageFile.LOAD_TRUNCATED_IMAGES = True


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def sliding_predict(model, image, num_classes, window_size=1024, flip=False):
    image_size = image.shape
    # print('image_size', image_size)
    h_s, w_s = image_size[2]/window_size, image_size[3]/window_size
    tile_size = (int(image_size[2] // h_s), int(image_size[3] // w_s))
    # print('tile_size', tile_size)
    overlap = 1 / 4

    stride = ceil(tile_size[0] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)

    # print(num_rows, num_cols)

    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            # print(row,col)
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            _, padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                _, fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions

    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask, output_path, image_file, palette):
    # Saves the image, the model output and the results after the post processing
    # w, h = image.size
    # image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    if colorized_mask.mode == "P":
        colorized_mask = colorized_mask.convert('RGB')
    colorized_mask.save(output_path)
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))


def mask2source(image_path, mask_path, output_path, alpha):
    img = cv2.imread(image_path)

    label = cv2.imread(mask_path,0)
    # print(np.unique(label))
    # print(mask_path)
    # print("[label]: ", np.unique(label), label.shape)
    # label = label[..., 0]
    # label[(label.any() != 3 and label.any() != 1)] = 0
    # label[label == 1] = 60
    # label[label == 3] = 180
    pd = np.zeros(img.shape)

    # print(np.unique(label))
    # label[label != 0] = 1
    pd[np.where(label == 0)] = [0, 0, 0]  # 黑色
    pd[np.where(label == 1)] = (0,255,255) # 青色
    pd[np.where(label == 2)] = [255,0,0] # 红色
    pd[np.where(label == 3)] = (0,255,0) # 绿色
    pd[np.where(label == 4)] = [0,0,255] # 蓝色
    pd[np.where(label == 5)] = [0,200,250] #深黄色
    pd[np.where(label == 6)] = [0,255,127]  # 深绿
    pd[np.where(label == 7)] = (255,0,255)  # 紫色

    pd = cv2.cvtColor(pd.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_merge = alpha * img + (1 - alpha) * pd

    cv2.imwrite(output_path, img_merge)


def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    # assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K']
    if dataset_type == 'CityScapes':
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    # print(checkpoint)

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
    # ===
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
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    print("Load model complete.>>>")
    x = torch.rand(1, 3, 512, 512).cuda()  # dummy data
    traced_script_module = torch.jit.trace(model, x)
    traced_script_module.save(r"/data/fpc/inference/weights/model.pt")
    print("torch.jit save model complete.>>>")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    #
    # torch.save(model, r'/data/fpc/output/outputs_10_28_3/river.pt')

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    print(os.path.join(args.images, f'*.{args.extension}'))
    # print(image_files)
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            file_name = img_file.split('/')[-1].split('.')[0] + args.suffix
            print(img_file)
            test = cv2.imread(img_file)
            print(test.shape)

            image = Image.open(img_file).convert('RGB')
            image_ = to_tensor(image)
            input = normalize(image_).unsqueeze(0)

            if args.mode == 'multiscale':
                prediction = multi_scale_predict(model, input, scales, num_classes, device)
            elif args.mode == 'sliding':
                # 除法的错误是这里的window_size设置的问题
                prediction = sliding_predict(model, input.cuda(), num_classes, window_size=256)
                # print('prediction', np.unique(prediction))
            else:
                prediction = model(input.to(device))
                prediction = prediction.squeeze(0).cpu().numpy()

            # 注： 分步存储的时候需要先对原prediction进行深拷贝

            # 生成彩色的图像供直接观察，前两类为黑白，二分类的时候自然就是黑白图像
            color_mask = prediction.copy()
            color_mask_output = os.path.join(args.output, 'color_mask_output')
            if not os.path.exists(color_mask_output):
                os.mkdir(color_mask_output)
            color_mask_output_path = os.path.join(color_mask_output, file_name)
            color_mask = F.softmax(torch.from_numpy(color_mask), dim=0).argmax(0).cpu().numpy()
            save_images(image, color_mask, color_mask_output_path, img_file, palette)

            # 存mask
            normal_mask_output = os.path.join(args.output, 'normal_mask_output')
            if not os.path.exists(normal_mask_output):
                os.mkdir(normal_mask_output)
            normal_mask_output_path = os.path.join(normal_mask_output, file_name)
            cv2.imwrite(normal_mask_output_path, color_mask)

            # 生成概率图，二分类的时候使用
            if args.probabilities:
                print('处理概率图')
                probabilities_mask = prediction.copy()
                probabilities_mask_output = os.path.join(args.output, 'probabilities_mask_output')
                if not os.path.exists(probabilities_mask_output):
                    os.mkdir(probabilities_mask_output)
                probabilities_mask_output_path = os.path.join(probabilities_mask_output, file_name)
                probabilities_mask = F.softmax(torch.from_numpy(probabilities_mask), dim=0).cpu().numpy() * 255

                # 输入的是w * h
                # (2, 26367, 29951) 这个是h, w
                # w = 29951, h = 26367
                probabilities_mask = probabilities_mask[1, :, :].astype(np.uint8)
                # print(np.unique(probabilities_mask))
                # print(probabilities_mask.shape)
                probabilities_mask[probabilities_mask > 50] = 255
                probabilities_mask[probabilities_mask <= 50] = 0
                cv2.imwrite(probabilities_mask_output_path, probabilities_mask)

            # maskOn
            maskOnRes_output = os.path.join(args.output, 'maskOnRes')
            if not os.path.exists(maskOnRes_output):
                os.mkdir(maskOnRes_output)
            maskOnRes_output_path = os.path.join(maskOnRes_output, file_name)
            mask2source(img_file, normal_mask_output_path, maskOnRes_output_path, 0.6)

            sys.exit()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_hrnet_road.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='sliding', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    # '/data/fpc/saved/HRNET_HDLOSS/road_second_0.3/best_model.pth' 道路
    #  '/data/fpc/saved/HRNET_HDLOSS/10-19_10-04/best_model.pth' 水体

    parser.add_argument('-m', '--model', default='/data/fpc/saved/HRNET_HDLOSS/road_second_0.3/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/data/deep/train/images', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='/data/fpc/output/outputs_12_13/', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-p', '--probabilities', default=False, type=bool,
                        help='generate probabilities prediction')
    parser.add_argument('-suf', '--suffix', default='.png', type=str,
                        help='output image type')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()