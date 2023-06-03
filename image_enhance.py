"""
亮度、对比度、饱和度、清晰度调整
"""
import os.path
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
    prediction_res_path = r'/data/fpc/output/outputs_09_21'
    output_file = os.path.join(r'/data/fpc/output/merge', prediction_res_path.split('/')[-1])
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    for img_name in os.listdir(prediction_res_path):
        img_path = os.path.join(prediction_res_path, img_name)
        image = cv2.imread(img_path)
        print(image.shape)

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
        image_max[image_max > 70] = 255
        image_max[image_max <= 70] = 0

        output_max_file = os.path.join(output_file, 'max')
        if not os.path.exists(output_max_file):
            os.mkdir(output_max_file)
        output_max_path = os.path.join(output_max_file, img_name)
        print(output_max_path)
        cv2.imwrite(output_max_path, image_max)
        # --------------mean-------------------------
        image_mean = np.mean(images_merge, axis=0)
        image_mean[image_mean > 90] = 255
        image_mean[image_mean <= 90] = 0

        output_min_file = os.path.join(output_file, 'min')
        if not os.path.exists(output_min_file):
            os.mkdir(output_min_file)
        output_min_path = os.path.join(output_min_file, img_name)
        print(output_min_path)
        cv2.imwrite(output_min_path, image_mean.astype(np.uint8))

        # source = '/data/fpc/inference/taibei/taibei.tif'
        # with rasterio.open(source) as raster_src:
        #     source_out = raster_src.read()
        #     raster_crs = raster_src.crs
        #     raster_transform = raster_src.transform
        # 
        # output_path = r'E:\stored_images\res_min_1.tif'
        # with rasterio.open(output_path, 'w', driver='GTiff',  # 图像类型
        #                    height=raster_src.height, width=raster_src.width,
        #                    count=1,  # 总层数
        #                    dtype=np.uint8,  # 数据类型
        #                    crs=raster_crs, transform=raster_transform) as dataset:
        #     dataset.write((mask_out[0]).astype(np.uint8))
