'''
Reference:
    https://github.com/d-li14/efficientnetv2.pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']

# effnetv2模型超参数
mode_cfgs = {
    "effnetv2_s": [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1]
    ],
    "effnetv2_m": [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1]
    ],
    "effnetv2_l": [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1]
    ],
    "effnetv2_xl": [
        # t, c, n, s, SE        # 256
        [1, 64, 4, 1, 0],       # 128
        [4, 96, 8, 2, 0],       # 64
        [4, 192, 8, 2, 0],      # 32
        [4, 192, 16, 2, 1],     # 16
        [6, 256, 24, 1, 1]      # 16
    ]
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se, use_checkpoint=False):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_checkpoint = use_checkpoint
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_checkpoint:
            x1 = checkpoint(self.conv, x)
        else:
            x1 = self.conv(x)

        if self.identity:
            return x + x1
        else:
            return x1


class UpBilinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBilinear, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        _diffY = x2.size()[2] - x1.size()[2]
        _diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [_diffX // 2, _diffX - _diffX // 2,
                        _diffY // 2, _diffY - _diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBilinearWithMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, t=4, c=96, n=8, use_se=0, use_checkpoint=False):
        super(UpBilinearWithMBConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        layer = []
        output_channel = _make_divisible(c * 1., 8)
        for i in range(n):
            if i+1==n:
                output_channel = out_channels
                # use_se = 1
            layer.append(MBConv(in_channels, output_channel, stride=1, expand_ratio=t, use_se=use_se, use_checkpoint=use_checkpoint))
            in_channels = output_channel
        self.conv = nn.Sequential(*layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        _diffY = x2.size()[2] - x1.size()[2]
        _diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [_diffX // 2, _diffX - _diffX // 2,
                        _diffY // 2, _diffY - _diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, in_channels=3, width_mult=1., mode="effnetv2_s", use_checkpoint=False):
        super(EffNetV2, self).__init__()
        assert mode in ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']
        self.cfgs = mode_cfgs[mode]

        # building first layer
        self.inc = DoubleConv(in_channels, 64)
        input_channel = _make_divisible(64 * width_mult, 8)
        # print("input_channel", input_channel)
        self.conv_1 = conv_3x3_bn(64, input_channel, 2)

        # building inverted residual blocks
        layers = []
        for t, c, n, s, use_se in self.cfgs:
            layer = []
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layer.append(MBConv(input_channel, output_channel, s if i == 0 else 1, t, use_se, use_checkpoint=use_checkpoint))
                input_channel = output_channel
            layers.append(nn.Sequential(*layer))

        self.features = nn.ModuleList(layers)

    def forward(self, x):
        inc = self.inc(x)
        x = self.conv_1(inc)
        outs = [inc]
        for num, blk in enumerate(self.features):
            x = blk(x)
            if num in [0, 1, 2, 4]:
                outs.append(x)
        return outs

def effnetv2_s(**kwargs):
    return EffNetV2(mode="effnetv2_s", **kwargs)


def effnetv2_m(**kwargs):
    return EffNetV2(mode="effnetv2_m", **kwargs)


def effnetv2_l(**kwargs):
    return EffNetV2(mode="effnetv2_l", **kwargs)


def effnetv2_xl(**kwargs):
    return EffNetV2(mode="effnetv2_xl", **kwargs)
