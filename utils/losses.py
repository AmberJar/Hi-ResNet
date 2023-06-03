import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight
from utils.lovasz_losses import lovasz_softmax
import math
from utils.hddtbinaryloss import HDDTBinaryLoss
from utils.semantic_connectivity_loss import SemanticConnectivityLoss


#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``:::::::fpc:::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2],
                                labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                      reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False,
                                           ignore_index=ignore_index,
                                           weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)
        self.OhemCELoss = OhemCELoss(thresh=0.3, n_min=12)
        self.gdice = GeneralizedDiceLoss()
        self.center = CenterLossN()

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)

        return CE_loss + dice_loss


class Focal_DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(Focal_DiceLoss, self).__init__()
        self.alpha = torch.tensor(
            [0.82574772, 0.83875163, 1.05331599, 1.26657997, 1, 0.75292588, 2.01820546, 0.78543563, 1.44473342,
             2.04551365, 0.97269181]).cuda()
        self.gdice = GeneralizedDiceLoss()
        self.focal = FocalLoss(alpha=self.alpha)
        self.gamma = 0.6

    def forward(self, output, target):
        gdice_loss = self.gdice(output, target)
        focoal_loss = self.focal(output, target)

        return gdice_loss + 0.8 * focoal_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class CE_LovaszLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255,
                 weight=None):
        super(CE_LovaszLoss, self).__init__()
        self.weight = [0.0635, 0.0645, 0.0810, 0.0974, 0.0769, 0.0579, 0.1552, 0.0604, 0.1111, 0.1573, 0.0748]
        self.lovasz = LovaszSoftmax()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(self.weight),
                                                 reduction=reduction,
                                                 ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        lovasz_loss = self.lovasz(output, target)

        return CE_loss + lovasz_loss


class SCELoss(nn.Module):
    def __init__(self, num_classes=2, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a  # 两个超参数
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)  # 最小设为 1e-4，即 A 取 -4
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class OhemCELoss(nn.Module):
    """

    Online hard example mining cross-entropy loss:在线难样本挖掘

    if loss[self.n_min] > self.thresh: 最少考虑 n_min 个损失最大的 pixel，

    如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，

    那么取实际所有大于该阈值的元素计算损失:loss=loss[loss>thresh]。

    否则，计算前 n_min 个损失:loss = loss[:self.n_min]

    """

    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()  # 将输入的概率 转换为loss值
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')  # 交叉熵

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)  # 排序
        if loss[self.n_min] > self.thresh:  # 当loss大于阈值(由输入概率转换成loss阈值)的像素数量比n_min多时，取所以大于阈值的loss值
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class CenterLossN(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=11, use_gpu=True):
        super(CenterLossN, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            N * C * H * W
        """
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)

        if self.use_gpu:
            centers = nn.Parameter(torch.randn(batch_size, self.num_classes, h, w)).cuda()
        else:
            centers = nn.Parameter(torch.randn(batch_size, self.num_classes, h, w))

        distmat = torch.pow(x, 2).cuda() + torch.pow(centers, 2)
        res = torch.zeros((distmat.size()))
        with torch.no_grad():
            for i, batch in enumerate(res):
                for j, _class in enumerate(batch):
                    res[i, j] = torch.addmm(input=distmat[i][j], mat1=x[i][j], mat2=centers[i][j], beta=1, alpha=-2)
        res_max = F.softmax(res, dim=1).max(dim=1)[0]
        # mask = labels.eq(labels)
        dist = res_max.cuda() * labels.float().cuda()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / (h * w * batch_size)

        return loss


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """
    def __init__(self, normalization='sigmoid', epsilon=1e-6, ignore_index=255):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.normalization = normalization
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=prediction.size()[1])
        output = F.softmax(prediction, dim=1)
        assert prediction.size() == target.size(), "'prediction' and 'target' must have the same shape"
        if prediction.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            prediction = torch.cat((prediction, 1 - prediction), dim=0)
            target = torch.cat((target, 1 - target), dim=0)
        prediction = torch.transpose(output, 1, 0)
        prediction = torch.flatten(prediction, 1) #flatten all dimensions except channel/class
        target = torch.transpose(target, 1, 0)
        target = torch.flatten(target, 1)
        target = target.float()
        w_l = target.sum(-1)
        w_l = 1 / (w_l ** 2).clamp(min=self.epsilon)
        w_l.requires_grad = False
        intersect = (prediction * target).sum(-1)
        intersect = intersect * w_l

        denominator = (prediction + target).sum(-1)
        # print(denominator)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        # print(denominator)
        return 1 - (2 * (intersect.sum() / denominator.sum()))


class LabelSmoothSoftmaxCE(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=255):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        # print(np.unique(label.cpu().numpy()))
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LSCE_GDLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(LSCE_GDLoss, self).__init__()
        self.gdice = GeneralizedDiceLoss(ignore_index=ignore_index)
        self.LSCE = LabelSmoothSoftmaxCE(ignore_index=ignore_index)
        self.hd_loss_func = HDDTBinaryLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        LSCE_loss = self.LSCE(output, target)
        gdice_loss = self.gdice(output, target)
        hd_loss = self.hd_loss_func(output, target)

        return LSCE_loss + gdice_loss + .15 * hd_loss


class LSCE_Connectivity_Loss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(LSCE_Connectivity_Loss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.LSCE = LabelSmoothSoftmaxCE()
        self.hd_loss = HDDTBinaryLoss()
        self.connectivity = SemanticConnectivityLoss()

    def forward(self, output, target):
        LSCE_loss = self.LSCE(output, target)
        connectivity_loss = self.connectivity(output, target)

        return LSCE_loss + 0.6 * connectivity_loss

from torch.nn.modules.loss import _Loss


def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)

class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index=-100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input, target) :
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        # self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        # if self.training and len(logits) == 2:
        #     logit_main, logit_aux = logits
        #     loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        loss = self.main_loss(logits, labels)

        return loss
