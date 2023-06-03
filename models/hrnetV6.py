import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hrnet_spatial_ocr_blockV6 import BNReLU, SpatialOCR_ASP_Module
from models.hrnet_backboneV6 import HRNetBackbone
from itertools import chain
from utils.helpers import set_trainable


class HRNet_W48_ASPOCR_V6(nn.Module):
    def __init__(self, num_classes, backbone, freeze_bn=False, freeze_backbone=False, **_):
        super(HRNet_W48_ASPOCR_V6, self).__init__()
        self.num_classes = num_classes
        self.backbone = HRNetBackbone(backbone)

        # extra added layers
        in_channels = 1080  # 48 + 96 + 192 + 384

        self.asp_ocr_head = SpatialOCR_ASP_Module(features=in_channels,
                                                  hidden_features=256,
                                                  out_features=256,
                                                  dilations=(24, 48, 72),
                                                  num_classes=self.num_classes,
                                                  bn_type=None)  # None

        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            BNReLU(512, bn_type=None),  # None
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)
        # feat5 = F.interpolate(x[4], size=(h, w), mode="bilinear", align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # coarse segmentation
        out_aux = self.aux_head(feats)

        # 计算内部点与整个object的相似度
        feats = self.asp_ocr_head(feats, out_aux)

        out = self.cls_head(feats)
        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out_aux, out

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.asp_ocr_head.parameters(), self.cls_head.parameters(), self.aux_head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    model = HRNet_W48_ASPOCR(num_classes=16, backbone='hrnet48')
    from torchinfo import summary

    summary(model, input_size=(6, 3, 256, 256))
    x_in = torch.rand(size=(6, 3, 256, 256)).cuda()
    outs = model(x_in)

    for out in outs:
        print(out.shape)