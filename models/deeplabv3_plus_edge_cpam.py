from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from utils.helpers import set_trainable
from itertools import chain
from torch.nn import Module, Conv2d,  Parameter, Softmax, Sequential, Dropout2d, ReLU, BatchNorm2d

# Position attention module
class PAM_Module(Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


# Channel attention module
class CAM_Module(Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CPAM_Module(Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(CPAM_Module, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p = Sequential(Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                 norm_layer(inter_channels),
                                 ReLU())
        self.conv_c = Sequential(Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                 norm_layer(inter_channels),
                                 ReLU())
        self.s_p = PAM_Module(inter_channels)
        self.s_c = CAM_Module(inter_channels)
        self.conv1_p = Sequential(Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                 norm_layer(inter_channels),
                                 ReLU())
        self.conv1_c = Sequential(Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                 norm_layer(inter_channels),
                                 ReLU())
        self.conv_out = Sequential(Dropout2d(0.1), Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv_p(x)
        sp_feat = self.s_p(feat1)
        sp_conv = self.conv1_p(sp_feat)

        feat2 = self.conv_c(x)
        sc_feat = self.s_c(feat2)
        sc_conv = self.conv1_c(sc_feat)

        feat_sum = sp_conv + sc_conv
        feat_output = self.conv_out(feat_sum)
        return feat_output



class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

''' 
-> ResNet BackBone
'''


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


''' 
-> (Aligned) Xception BackBone
Pretrained model from https://github.com/Cadene/pretrained-models.pytorch
by Remi Cadene
'''


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU(inplace=True)

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                nn.BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()

        # Stride for block 3 (entry flow), and the dilation rates for middle flow and exit flow
        if output_stride == 16: b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Entry Flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)

        # Middle Flow
        for i in range(16):
            exec(f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)')

        # Exit flow
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0], exit_flow=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = nn.BatchNorm2d(2048)

        initialize_weights(self)
        if pretrained: self._load_pretrained_model()

    def _load_pretrained_model(self):
        url = 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
        pretrained_weights = model_zoo.load_url(url)
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)  # [C, C] -> [C, C, 1, 1]
                if k.startswith('block11'):
                    # In Xception there is only 8 blocks in Middle flow
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
        x = self.block2(x)
        x_8 = x
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_features, x_8


''' 
-> The Atrous Spatial Pyramid Pooling
'''


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


''' 
-> Decoder
'''


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


'''
-> Deeplab V3 +
'''


class DeepLabEdgeCPAM(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True,
                 output_stride=16, freeze_bn=False, freeze_backbone=False, **_):

        super(DeepLabEdgeCPAM, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.CPAM = CPAM_Module(in_channels=2048, out_channels=256, norm_layer=BatchNorm2d)
        self.conv_assp_cpam = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2)
        )
        self.decoder = Decoder(low_level_channels, num_classes)

        sp8_inplanes = 256
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features, x_8 = self.backbone(x)

        # use_boundary_8
        feat_out_sp8 = self.conv_out_sp8(x_8)

        x_assp = self.ASSP(x)
        x_cpam = self.CPAM(x)
        x = torch.cat([x_assp, x_cpam], dim=1)
        x = self.conv_assp_cpam(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x, feat_out_sp8

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    a = torch.randn(size=(2, 3, 256, 256))
    model = DeepLabEdgeCPAM(num_classes=2)
    b= model(a)
    print("========================")
    for ele in b:
        print(ele.shape)