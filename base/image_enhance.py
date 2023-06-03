"""
亮度、对比度、饱和度、清晰度调整
"""

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

    # 亮度调整
    imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)

    # 对比度调整
    contrastEnhancer = ImageEnhance.Contrast(x)
    x = contrastEnhancer.enhance(contrast)

    # 饱和度调整
    colorEnhancer = ImageEnhance.Color(x)
    x = colorEnhancer.enhance(color)

    # 清晰度调整
    SharpnessEnhancer = ImageEnhance.Sharpness(x)
    x = SharpnessEnhancer.enhance(sharpness)

    x = img_to_array(x)
    return x


def random_enhance(x, brightness_range=(.7, 1.3), contrast_range=(.7, 1.3), color_range=(.7, 1.3), sharpness_range=(.7, 1.3)):
    if len(brightness_range) != 2:
        raise ValueError(
            '`brightness_range should be tuple or list of two floats. '
            'Received: %s' % (brightness_range,))

    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    color = np.random.uniform(color_range[0], color_range[1])
    sharpness = np.random.uniform(sharpness_range[0], sharpness_range[1])

    return apply_enhance_shift(x, brightness=brightness, contrast=contrast, color=color, sharpness=sharpness)
