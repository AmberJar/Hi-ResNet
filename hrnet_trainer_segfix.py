import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from utils.hddtbinaryloss import HDDTBinaryLoss
from utils.semantic_connectivity_loss import SemanticConnectivityLoss
import random
import cv2
import torch.nn.functional as F
from utils.losses import CenterLossN
import os
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from utils.TokenMix import Mixup
from utils.segfixloss import SegFixLoss
# Add.60%的batch会将四张图拼成一张图.batch_size为4的倍数


def concat_tensor(tensor_in):
    if tensor_in.shape[0] % 4 == 0:
        tensor_in_0 = tensor_in[0::4, :, :, :]
        tensor_in_1 = tensor_in[1::4, :, :, :]
        tensor_in_2 = tensor_in[2::4, :, :, :]
        tensor_in_3 = tensor_in[3::4, :, :, :]

        tensor_in_01 = torch.cat([tensor_in_0, tensor_in_1], dim=2)
        tensor_in_23 = torch.cat([tensor_in_2, tensor_in_3], dim=2)
        tensor_in = torch.cat([tensor_in_01, tensor_in_23], dim=3)
    return tensor_in


class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader=None, val_loader=None, local_rank=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']:self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor()])

        if self.device == torch.device('cpu'): prefetch = False
        # if prefetch:
        #     self.train_loader = DataPrefetcher(train_loader, device=self.device)
        #     self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
        # self.model.backbone.trainable = False
        self.model.train()

        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()

        # 初始化segfix loss
        segfixfunc = SegFixLoss()

        # 仅rank=0的时候显示tqdm
        if self.local_rank == 0:
            tbar = tqdm(self.train_loader, ncols=150)
        else:
            tbar = self.train_loader

        for batch_idx, (data, target, distance_map, angle_map) in enumerate(tbar):
            data = data.to(self.device)
            target = target.to(self.device)
            distance_map = distance_map.to(self.device)
            angle_map = angle_map.to(self.device)

            self.data_time.update(time.time() - tic)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()

            r = np.random.rand(1)

            # beta，迪利克雷分布参数，正反硬币次数，整数
            # cutmix_prob，cutmix的概率
            cutmix_prob = 0.5
            beta = 1

            if beta > 0 and r < cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(data.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                out_aux, output, mask_map, dir_map = self.model(data)
                # compute output
                loss = self.loss(output, target_a) * lam + self.loss(output, target_b) * (1. - lam)
                loss_aux = self.loss(out_aux, target_a) * lam + self.loss(out_aux, target_b) * (1. - lam)
                loss1 = loss + .4 * loss_aux
            else:
                # compute output
                out_aux, output, mask_map, dir_map = self.model(data)
                loss = self.loss(output, target)
                loss_aux = self.loss(out_aux, target)
                loss1 = loss + .4 * loss_aux

            mask_loss, direction_loss = segfixfunc([mask_map, dir_map], [target, distance_map, angle_map])

            if isinstance(self.loss, torch.nn.parallel.DistributedDataParallel):
                loss1 = loss.mean()

            if isinstance(self.loss, torch.nn.parallel.DistributedDataParallel):
                mask_loss = mask_loss.mean()
                direction_loss = direction_loss.mean()

            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self.num_classes

            segfix_group_loss = mask_loss + direction_loss
            losses = loss1 + 0.1 * segfix_group_loss
            losses.backward()
            self.optimizer.step()

            if self.ema:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.ema.update(name, param.data)

            self.total_loss.update(losses.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if self.local_rank == 0:
                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                    self.writer.add_scalar(f'{self.wrt_mode}/loss', losses.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, Class_IoU = self._get_seg_metrics().values()

            # PRINT INFO
            if self.local_rank == 0:
                tbar.set_description(
                    'TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} | {} | {:.3f} | {:.3f} |'.format(
                        epoch, self.total_loss.average,
                        pixAcc, mIoU,
                        self.batch_time.average, self.data_time.average, Class_IoU, loss1.item(), segfix_group_loss.item()))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        if self.local_rank == 0:
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
                # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {
            'loss': self.total_loss.average,
            **seg_metrics
        }

        self.lr_scheduler.step()  # pytorch version > 1.1.0
        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()

        if self.local_rank == 0:
            tbar = tqdm(self.val_loader, ncols=150)
        else:
            tbar = self.val_loader

        segfixfunc = SegFixLoss()
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target, distance_map, angle_map) in enumerate(tbar):  # [0,1,0,1]
                data, target, distance_map, angle_map = data.to(self.device), target.to(self.device), distance_map.to(self.device), angle_map.to(self.device)

                # LOSS
                out_aux, output, mask_map, dir_map = self.model(data)
                loss = self.loss(output, target)
                loss_aux = self.loss(out_aux, target)

                mask_loss, direction_loss = segfixfunc([mask_map, dir_map], [target, distance_map, angle_map])

                if isinstance(self.loss, torch.nn.parallel.DistributedDataParallel):
                    loss = loss.mean()
                    loss_aux = loss_aux.mean()
                    mask_loss = mask_loss.mean()
                    direction_loss = direction_loss.mean()

                loss1 = loss + .4 * loss_aux
                losses = loss1 + 0.1 * (mask_loss + direction_loss)

                self.total_loss.update(losses.item())
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                if self.local_rank == 0:
                    tbar.set_description(
                        'EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(
                            epoch, self.total_loss.average, pixAcc, mIoU
                        )
                    )
                # if batch_idx%100==0:
                #     print('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch, self.total_loss.average, pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            if self.local_rank == 0:
                self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            seg_metrics = self._get_seg_metrics()
            if self.local_rank == 0:
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
