# coding=utf-8
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import torch
from torchvision import transforms
import dataloaders
import argparse
import json
import models
from collections import OrderedDict
import time
from tqdm import tqdm
tic = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0， 1， 2， 3"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

window_size = 512

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_hrnet_cpam_hdloss.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default=r'/data/fpc/saved/HRNET_HDLOSS/10-14_10-59/checkpoint-epoch59.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/chenyuxia/16_taibei/taibei', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default=r'/data/chenyuxia/outputs/outputs_10_17', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


args = parse_arguments()
config = json.load(open(args.config))

to_tensor = transforms.ToTensor()
# loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
normalize = transforms.Normalize([0.486128, 0.452417, 0.491495], [0.154808, 0.140131, 0.147835])
num_classes = 2

# Model
print("Load model ............")
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
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
    print(images.shape)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    w, h, c = images.shape
    mask_steps = np.zeros((w, h, 2))
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
                preds = torch.softmax(preds, dim=1).detach().cpu().numpy()

            outs = [ele for ele in preds]
            outs = np.concatenate(outs, axis=-1)
            slices_outs.append(outs)

        slices_outs = np.concatenate(slices_outs, axis=1)
        slices_outs = slices_outs.transpose([1, 2, 0])
        print("slices_outs.shape", slices_outs.shape)
        slices_outs = slices_outs[cuts:cuts + w, cuts: cuts + h, :]
        # cv2.imwrite('/data/chenyuxia/road_text' + path[len(args.images): -len(args.extension)] + "tif", (slices_outs[:, :, 1] * 255).astype(np.uint8))
        print('here')
        print(slices_outs.shape)
        mask_steps += slices_outs

    mask_steps = mask_steps / len(CUTS_PNT)
    mask_steps = mask_steps[:, :, 1]
    print('here too')
    print(mask_steps.shape)
    return mask_steps


from glob import glob

files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
print(files)

tbar = tqdm(files, ncols=100)
i = 0
for path in tbar:
    print(path)
    if i >= 23:
        break
    outs = predicts(path, model)
    i += 1
    outs = outs * 255
     # outs[outs < 50] = 0
     # outs[outs >= 50] = 255
    save_path = r'/data/chenyuxia/outputs/outputs_10_17' + path[len(args.images): -len(args.extension)] + "tif"
    print(save_path)
    cv2.imwrite(save_path, outs.astype(np.uint8))
toc = time.time()
print(toc - tic)
"""
亮度、对比度、饱和度、清晰度调整
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import random
from glob import glob
import cv2
import numpy as np

try:
    import scipy
    from scipy import ndimage
except ImportError:
    scipy = None

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

__doc__ = [
    '''random_enhance(
    x, 
    brightness_range=(.7, 1.3), 
    contrast_range=(.7, 1.3), 
    color_range=(.7, 1.3), 
    sharpness_range=(.7, 1.3)
    )'''
]


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def apply_enhance_shift(x, brightness=1., contrast=1., color=1., sharpness=1.):
    if ImageEnhance is None:
        raise ImportError('Using image enhance requires PIL. Install PIL or Pillow.')
    x = array_to_img(x)

    funcs = ["ImageEnhance.Brightness(x).enhance(brightness)",
             "ImageEnhance.Contrast(x).enhance(contrast)",
             "ImageEnhance.Color(x).enhance(color)",
             "ImageEnhance.Sharpness(x).enhance(sharpness)"]
    random.shuffle(funcs)

    for fun in funcs:
        if random.random() > .35:
            x = eval(fun)

    x = img_to_array(x)
    return x


def random_enhance(x, brightness_range=(.7, 1.3), contrast_range=(.7, 1.3), color_range=(.7, 1.3),
                   sharpness_range=(.7, 1.3)):
    if len(brightness_range) != 2:
        raise ValueError(
            '`brightness_range should be tuple or list of two floats. '
            'Received: %s' % (brightness_range,))

    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    color = np.random.uniform(color_range[0], color_range[1])
    sharpness = np.random.uniform(sharpness_range[0], sharpness_range[1])

    return apply_enhance_shift(x, brightness=brightness, contrast=contrast, color=color, sharpness=sharpness)


if __name__ == '__main__':
    prediction_res_path = r'/data/chenyuxia/roads_show'
    output_file = r'/data/chenyuxia/roads_show/predicts'
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    for img_name in os.listdir(prediction_res_path):
        img_path = os.path.join(prediction_res_path, img_name)
        print(img_path)
        image = cv2.imread(img_path)

        rot_res = []  # 儲存rotation的結果變量
        names = locals()  # 動態命名空間
        for i in range(3):
            print(i)
            img_copy = random_enhance(image)
            print(img_copy.shape)

            names['rot_{}_90'.format(str(i))] = np.rot90(img_copy, 1).astype(np.uint8)
            names['rot_{}_180'.format(str(i))] = np.rot90(img_copy, 2).astype(np.uint8)
            names['rot_{}_270'.format(str(i))] = np.rot90(img_copy, 3).astype(np.uint8)

            rot_res.append(names['rot_{}_90'.format(str(i))])
            rot_res.append(names['rot_{}_180'.format(str(i))])
            rot_res.append(names['rot_{}_270'.format(str(i))])

        image_list = []
        for num, pic in enumerate(rot_res):
            print("============================>")
            print(num, pic.shape)

            if num % 3 == 0:
                pic = np.rot90(pic, 3)
                print("rotate 1", pic.shape)
            elif num % 3 == 1:
                pic = np.rot90(pic, 2)
                print("rotate 2", pic.shape)
            elif num % 3 == 2:
                pic = np.rot90(pic, 1)
                print("rotate 3", pic.shape)
            print(np.unique(pic))
            pic = np.expand_dims(pic, 0)
            image_list.append(pic)

        images_merge = np.concatenate(image_list, axis=0)

        # ---------------------max-----------------------
        image_max = np.max(images_merge, axis=0)

        output_max_file = os.path.join(output_file, 'max')
        if not os.path.exists(output_max_file):
            os.mkdir(output_max_file)
        output_max_path = os.path.join(output_max_file, img_name)
        print(output_max_path)
        output_max_pro_path = os.path.join(output_max_file, 'pro_' + img_name)
        cv2.imwrite(output_max_pro_path, image_max.astype(np.uint8))
        image_max[image_max > 25] = 255
        image_max[image_max <= 25] = 0
        cv2.imwrite(output_max_path, image_max.astype(np.uint8))
        # --------------mean-------------------------
        image_mean = np.mean(images_merge, axis=0)

        output_mean_file = os.path.join(output_file, 'mean')
        if not os.path.exists(output_mean_file):
            os.mkdir(output_mean_file)
        output_mean_path = os.path.join(output_mean_file, img_name)
        print(output_mean_path)
        output_mean_pro_path = os.path.join(output_mean_file, 'pro_' + img_name)
        cv2.imwrite(output_mean_pro_path, image_mean.astype(np.uint8))
        image_mean[image_mean > 25.5] = 255
        image_mean[image_mean <= 25] = 0
        cv2.imwrite(output_mean_path, image_mean.astype(np.uint8))

import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg


def union_image_mask(image_path, mask_path, num):
    # 读取原图
    image = cv2.imread(image_path)
    print(image.shape)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 裁剪到和原图一样大小
    print(mask_2d.shape)
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)
    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    cv2.imwrite(os.path.join("/data/chenyuxia/outputs/res_roads", str(num) + ".bmp"), image)


def mask2source(image_path, mask_path, output_path, alpha):
    # root_path_background = "/home/cc/codes/python/YOLOP-main/datas/images/train/"
    # root_path_paste = "/home/cc/codes/python/YOLOP-main/datas/da_seg_annotations/train/"
    # output_path = "/home/cc/codes/python/YOLOP-main/datas/masks/"
    # img_list = os.listdir(root_path_background)
    # label_list = os.listdir(root_path_paste)
    # img_list = sorted(img_list)
    # label_list = sorted(label_list)
    # for num, img_label in enumerate(zip(img_list, label_list)):
    img = cv2.imread(image_path)
    label = cv2.imread(mask_path, 0)
    # label[(label.any() != 3 and label.any() != 1)] = 0
    # label[label == 1] = 60
    # label[label == 3] = 180
    pd = np.zeros(img.shape)
    pd[np.where(label == 0)] = [0, 0, 0] #黑
    pd[np.where(label == 255)] = [127, 255, 0]  # 绿
    # pd[np.where(label == 2)] = [184, 134, 11]  # 棕色
    # pd[np.where(label == 3)] = [0, 0, 205] # Mediumblue
    # pd[np.where(label == 4)] = [255, 20, 147] # deep pink
    # pd[np.where(label == 5)] = [178, 34, 34]  # Firebrick
    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    img_merge = alpha * img + (1 - alpha) * pd
    cv2.imwrite(output_path, img_merge)


if __name__ == '__main__':
    mask_pth = '/data/chenyuxia/roads_show/predicts/mean'
    img_path = '/data/chenyuxia/16_taibei/taibei_splited_16'
    out_path = '/data/chenyuxia/outputs/res_roads'
    for i in os.listdir(img_path):
        input_image = os.path.join(img_path, i)
        j = i.split('.')[0] + '.png'
        print(j)
        input_mask = os.path.join(mask_pth, j)
        output = os.path.join(out_path, i)
        mask2source(input_image, input_mask, output, 0.6)
