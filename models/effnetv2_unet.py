'''
Reference:
    https://github.com/d-li14/efficientnetv2.pytorch

Unet ---->
Up1 up torch.Size([2, 512, 16, 16]) en torch.Size([2, 512, 32, 32])
Up2 up torch.Size([2, 256, 32, 32]) en torch.Size([2, 256, 64, 64])
Up3 up torch.Size([2, 128, 64, 64]) en torch.Size([2, 128, 128, 128])
Up4 up torch.Size([2, 64, 128, 128]) en torch.Size([2, 64, 256, 256])
Out torch.Size([2, 64, 256, 256])
'''


import math
import torch
import torch.nn as nn
from base import BaseModel
from models import effnetv2
from utils.helpers import set_trainable
from models.effnetv2 import mode_cfgs, UpBilinear, UpBilinearWithMBConv, OutConv
from itertools import chain


def _getfilter_from_mode_cfgs(mode="effnetv2_xl"):
    filters = [64]
    for t, c, n, s, SE in mode_cfgs[mode]:
        if s==2 and SE==1:
            pass
        else:
            filters.append(c)
    filters = filters[::-1]
    out_shape = []
    for i in range(len(filters)-1):
        out_shape.append((filters[i] + filters[i+1], filters[i+1]))
    return out_shape


class EffNetV2Unet(BaseModel):
    def __init__(self, n_classes=2, in_channels=3, width_mult=1.,
                 backbone="effnetv2_xl", with_MBConv=True, use_checkpoint=False,
                 expand_ratio=[2, 2, 2, 2], n_iters=[2, 2, 2, 2],
                 freeze_bn=False, freeze_backbone=False
                 ):
        super(EffNetV2Unet, self).__init__()
        self.down_feature = getattr(effnetv2, backbone)(in_channels=in_channels, width_mult=width_mult)
        self.up_filters = _getfilter_from_mode_cfgs(backbone)
        if with_MBConv:
            self.up1 = UpBilinearWithMBConv(self.up_filters[0][0], self.up_filters[0][1], t=expand_ratio[0], c=self.up_filters[0][0], n=n_iters[0], use_checkpoint=use_checkpoint)
            self.up2 = UpBilinearWithMBConv(self.up_filters[1][0], self.up_filters[1][1], t=expand_ratio[1], c=self.up_filters[1][0], n=n_iters[1], use_checkpoint=use_checkpoint)
            self.up3 = UpBilinearWithMBConv(self.up_filters[2][0], self.up_filters[2][1], t=expand_ratio[2], c=self.up_filters[2][0], n=n_iters[2], use_checkpoint=use_checkpoint)
            self.up4 = UpBilinearWithMBConv(self.up_filters[3][0], self.up_filters[3][1], t=expand_ratio[3], c=self.up_filters[3][0], n=n_iters[3], use_checkpoint=use_checkpoint)
        else:
            self.up1 = UpBilinear(self.up_filters[0][0], self.up_filters[0][1])
            self.up2 = UpBilinear(self.up_filters[1][0], self.up_filters[1][1])
            self.up3 = UpBilinear(self.up_filters[2][0], self.up_filters[2][1])
            self.up4 = UpBilinear(self.up_filters[3][0], self.up_filters[3][1])
        self.outc = OutConv(64, n_classes)

        self._initialize_weights()
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.down_feature(x)

        # print("Up1", "up", x5.shape, "en", x4.shape)
        x = self.up1(x5, x4)
        # print("Up2", "up", x.shape, "en", x3.shape)
        x = self.up2(x, x3)
        # print("Up3", "up", x.shape, "en", x2.shape)
        x = self.up3(x, x2)
        # print("Up4", "up", x.shape, "en", x1.shape)
        x = self.up4(x, x1)
        # print("Out", x.shape)
        logits = self.outc(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def get_backbone_params(self):
        return self.down_feature.parameters()

    def get_decoder_params(self):
        return chain(self.up1.parameters(), self.up2.parameters(),
                     self.up3.parameters(), self.up4.parameters(),
                     self.outc.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    from torchinfo import summary
    import numpy as np
    a = np.random.uniform(-1, 1, (4, 3, 256, 256)).astype(np.float32)
    x_in = torch.from_numpy(a)
    model = EffNetV2Unet(backbone="effnetv2_xl", n_classes=2, with_MBConv=True).cuda()

    summary(model, input_size=(3, 3, 256, 256))
    model.eval()
    with torch.no_grad():
        x_out = model(x_in.cuda())
    print(x_out.shape)
