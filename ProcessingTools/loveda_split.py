import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/mnt/data/chenyuxia/EXPERIMENTAL/loveda/raw_data/images")
    parser.add_argument("--mask-dir", default="/mnt/data/chenyuxia/EXPERIMENTAL/loveda/raw_data/labels")
    parser.add_argument("--output-img-dir", default="/mnt/data/chenyuxia/EXPERIMENTAL/loveda/seed0/train/images")
    parser.add_argument("--output-mask-dir", default="/mnt/data/chenyuxia/EXPERIMENTAL/loveda/seed0/train/labels")
    parser.add_argument("--rgb-image", default=True)  # use Potsdam RGB format images
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)  # ignore
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad
    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=img)
    if mode == 'train':
        pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=img)

    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=mask)
    img_pad, mask_pad = pad_img['image'], pad_mask['image']
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad




def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)

        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())


        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size, mode)

            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)

        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list



def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir,  rgb_image,
     mode, val_scale, split_size, stride) = inp
    img_filename = os.path.basename(img_path).split('.')[0]
    mask_filename = os.path.basename(mask_path).split('.')[0]
    # print(eroded)
    print(img_filename)
    # print(mask_filename)
    # mask_path = mask_path + '.png'
    # img_path = img_path + '.png'
    img = Image.open(img_path).convert('RGB')
    # print(img)
    mask = Image.open(mask_path).convert('RGB')

    # print(mask)
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          val_scale=val_scale, mode=mode)
    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile_cut = img[y:y + split_size, x:x + split_size]
                mask_tile_cut = mask[y:y + split_size, x:x + split_size]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:

                    out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(img_filename, m, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    rgb_image = args.rgb_image
    mode = args.mode
    print(mode)
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths_raw = glob.glob(os.path.join(imgs_dir, "*.png"))
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.png"))
    img_paths_raw.sort()
    mask_paths_raw.sort()
    # print(img_paths[:10])
    # print(mask_paths[:10])

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, rgb_image,
            mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths_raw, mask_paths_raw)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


