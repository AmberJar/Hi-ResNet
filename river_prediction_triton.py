import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from math import ceil
from PIL import Image
from utils.helpers import colorize_mask
from PIL import ImageFile
import rasterio
import gc
import ssl
import tritonclient.grpc as grpc_client
ssl._create_default_https_context = ssl._create_unverified_context
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
CLIENT = grpc_client.InferenceServerClient(url="192.168.128.29:8001")
import os

def adjust_pad(ele, window_size):
    if ele < 0:
        return ele + window_size
    else:
        return ele


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def sliding_predict(model, image, num_classes, func, flip=False):
    image_size = image.shape
    print('image_size', image_size)
    tile_size = (int(image_size[2] // 8), int(image_size[3] // 8))
    print('tile_size', tile_size)
    overlap = 1 / 2

    stride = ceil(tile_size[0] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    sum_num = num_rows * num_rows

    print(num_rows, num_cols)

    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            print(row, col)
            func((row * (num_rows - 1) + col) / sum_num)
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
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save('/data/chenyuxia/outputs/GDL11NAIC/standard_taibei_36.png')
    return colorized_mask

# CLIENT.infer ---->models替换
# img -----> inputs
to_tensor = transforms.ToTensor()
# normalize = transforms.Normalize(loader.MEAN, loader.STD)

num_classes = 2
normalize = transforms.Normalize([0.486128, 0.452417, 0.491495], [0.154808, 0.140131, 0.147835])
def predict_batchsize(inputs, model_name = 'building', batchsize=64,in_channels="first",out_channels="first",inp_desc=("INPUT__0","FP32"),otp_desc=("OUTPUT__0","FP32")):

    outs = []
    b, c, w, h = inputs.shape
    b_1st = b % batchsize
    cnt_range = b // batchsize
    img = inputs
    ## 数据准备
    if inp_desc[1] == "FP32":
        img = img.astype("float32")
    elif inp_desc[1] == "UINT8":
        img = img.astype("uint8")
    else:
        pass

    otp = grpc_client.InferRequestedOutput(otp_desc[0])

    if b_1st > 0:
        # with torch.no_grad():
        temp_data = inputs[:b_1st, ...]
        print("temp_data", type(temp_data), temp_data.shape)
        inp = grpc_client.InferInput(inp_desc[0], temp_data.shape, inp_desc[1])
        inp.set_data_from_numpy(temp_data)
        preds = CLIENT.infer(model_name=model_name, inputs=[inp], outputs=[otp]).as_numpy(otp_desc[0])
        outs.append(preds)


    for iter in range(0, cnt_range):
        temp_data = img[b_1st + iter * batchsize:b_1st + (iter + 1) * batchsize, ...]
        print("temp_data", type(temp_data), temp_data.shape)
        inp = grpc_client.InferInput(inp_desc[0], temp_data.shape, inp_desc[1])
        inp.set_data_from_numpy(temp_data)
        preds = CLIENT.infer(model_name=model_name, inputs=[inp], outputs=[otp]).as_numpy(otp_desc[0])
        outs.append(preds)
    outs = np.concatenate(outs, axis=0)  # numpy 1.23.3

    return outs

def predicts_map(image_path, num_classes, model_name, func, window_size=256, step_slice=32):
    to_tensor = transforms.ToTensor()
    pad_center = int(window_size * .25 / 2)  # 可自由调节

    with rasterio.open(image_path) as raster_src:
        data = raster_src.read()

    data = np.transpose(data, (2, 1, 0))  # RGB, --> w,h,c
    w, h, c = data.shape

    mask_steps = np.zeros((w, h, num_classes))
    CUTS_PNT = [x for x in range(0, window_size, step_slice)]

    # 统计进度
    for idx_step, cuts in enumerate(CUTS_PNT):
        print(cuts, CUTS_PNT)
        datas = data.copy()
        images_copy = np.pad(datas, ((cuts, adjust_pad(window_size - w % window_size - cuts, window_size)),
                                     (cuts, adjust_pad(window_size - h % window_size - cuts, window_size)),
                                     (0, 0)
                                     ), mode="constant", constant_values=0)

        nw, nh, c = images_copy.shape
        images_copy = np.pad(images_copy, ((pad_center, pad_center), (pad_center, pad_center), (0, 0)), mode="constant",
                             constant_values=0)

        # 分割 + 重建
        slices_outs = []
        width_seqs = [x for x in range(0 + pad_center, nw + pad_center, window_size)]
        for idx_width, st_w in enumerate(width_seqs):
            batch_data = []

            func( ((idx_width + 1) / len(width_seqs) + idx_step) / len(CUTS_PNT) )

            for st_h in range(0 + pad_center, nh + pad_center, window_size):
                tmp_img_0 = images_copy[
                            st_w - pad_center: st_w + window_size + pad_center,
                            st_h - pad_center: st_h + window_size + pad_center, :3]

                tmp_img = normalize(to_tensor(tmp_img_0)).unsqueeze(0)
                batch_data.append(tmp_img)

            inputs = np.concatenate(batch_data)

            preds = predict_batchsize(inputs=inputs, model_name=model_name, batchsize=64, in_channels="first",out_channels="first",inp_desc=("INPUT__0","FP32"),otp_desc=("OUTPUT__0","FP32"))
            preds = torch.from_numpy(preds)
            preds = torch.softmax(preds, dim=1).detach().cpu().numpy()
            preds = preds[:, :, pad_center:-pad_center, pad_center:-pad_center]
            outs = [ele for ele in preds]


            outs = np.concatenate(outs, axis=-1)
            slices_outs.append(outs)

        slices_outs = np.concatenate(slices_outs, axis=1)
        slices_outs = slices_outs.transpose([1, 2, 0])
        slices_outs = slices_outs[cuts:cuts + w, cuts: cuts + h, :]
        mask_steps += slices_outs

    del slices_outs, data, images_copy
    gc.collect()

    mask_steps = mask_steps / len(CUTS_PNT)   # W, H, C
    # mask_steps = mask_steps[:, :, 1]
    return mask_steps

def Inference_Building(img_path, func, x):
    '''
       :param img_path: 需要处理的图片路径
       :param func: 进度条回调函数
       :param class_type: build/ river。。。
       :return: ndarray
       '''
    palette = palette_generation(x)
    model_name = 'roads'
    # 加载模型
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda' if len(availble_gpus) > 0 else 'cpu')
    # if x == "建筑预测":
    model = torch.load(f="/data/chenyuxia/road_model.pt", map_location=device)
    print('model成功导入')

    if "cuda" in device.type:
        pass
       # model = torch.nn.DataParallel(model)
    # model.to(device)
    # model.eval()

    prediction = predicts_map(image_path=img_path, num_classes=num_classes, model_name=model_name, func=func, window_size=512,
                              step_slice=32).transpose(1, 0)
    print(f"|------------->{prediction.shape}")
    m = 0
    prediction = np.argmax(prediction, axis=-1)

    result = save_images(
        None,
        prediction,
        os.path.join('/data/liyonggang/hemu/project/adaspace_aerial_image_to_2dmaps/streamlit_/predicted'),
        img_path,
        palette)
    print('成功保存文件')
    return result


def palette_generation(x):
    if x == '背景':
        AdaSpaceMaps_palette = [0, 0, 0,  # 背景
                                255, 182, 193,  # 裸土
                                255, 182, 193,  # 矮植被
                                255, 182, 193,  # 高植被
                                255, 182, 193,  # 河流
                                255, 182, 193
                                ]
    elif x == '裸土':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                89, 61, 59,  # 裸土
                                255, 182, 193,  # 矮植被
                                255, 182, 193,  # 高植被
                                255, 182, 193,  # 河流
                                255, 182, 193
                                ]

    elif x == '矮植被':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  # 裸土
                                3, 140, 101,  # 矮植被
                                255, 182, 193,  # 高植被
                                255, 182, 193,  # 河流
                                255, 182, 193
                                ]
    elif x == '高植被':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  # 裸土
                                255, 182, 193,  # 矮植被
                                3, 140, 127,  # 高植被
                                255, 182, 193,  # 河流
                                255, 182, 193
                                ]
    elif x == '河流':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  # 裸土
                                255, 182, 193,  # 矮植被
                                255, 182, 193,  # 高植被
                                126, 208, 248,  # 河流
                                255, 182, 193
                                ]

    elif x == '建筑':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  # edge
                                129, 72,  105 # build
                                ]

    elif x=='边缘':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                51, 51, 51,  # edge
                                255, 182, 193  # build
                                ]
    elif x =='道路':
        AdaSpaceMaps_palette = [255, 182, 193,  #
                                0, 225, 0,  # road
                                ]
    elif x == '植被':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,
                                255, 182, 193,
                                255, 182, 193,
                                3, 140, 101,
                                3, 140, 127,
                                255, 182, 193
                                ]

    elif x== '水体':
        AdaSpaceMaps_palette = [255, 182, 193,  # 背景
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,  #
                                255, 182, 193,
                                255, 182, 193,
                                88, 195, 224
                                ]

    return AdaSpaceMaps_palette

# 原始预测文件


if __name__ == '__main__':
    Inference_Building('/data/chenyuxia/16_taibei/taibei/standard_taibei_36_001.png', print, '道路')