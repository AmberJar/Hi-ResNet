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
# from utils.losses import CELoss
from utils.hddtbinaryloss import HDDTBinaryLoss
from utils.semantic_connectivity_loss import SemanticConnectivityLoss
# from utils.losses import CenterLoss
import random
import torch.nn.functional as F


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
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes
        self.batch_size = self.train_loader.batch_size

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor()])

        if self.device == torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

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
        tbar = tqdm(self.train_loader, ncols=130)

        # connectivity loss
        semantic_loss_func = SemanticConnectivityLoss(alpha=1, beta=0.3)

        for batch_idx, (data, target) in enumerate(tbar):
            # print('batch_idx', batch_idx)
            # print('data', data.shape)
            # print('target',  target.shape)

            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)
            # self.lr_scheduler.step(epoch=epoch-1) # pytorch version==1.1.0
            if batch_idx % 100 < 50:
                data = concat_tensor(data)
                target = concat_tensor(target.unsqueeze(1)).squeeze(1)
            # Switch Channel
            # if random.uniform(0, 1) > .8:
            #     data = torch.cat(
            #         [data[:, 2:3, :, :], data[:, 1:2, :, :], data[:, 0:1, :, :]], dim=1
            #     )

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            out_aux, output = self.model(data)

            '''
            hd_loss = hd_loss_func(output, target)
            hd_loss_aux = hd_loss_func(out_aux, target)
            '''

            theta = 0.4

            # se_connectivity_loss = semantic_loss_func(output, target)
            # se_connectivity_loss_aux = semantic_loss_func(out_aux, target)

            # center_loss = center_loss_func(output, target)
            # center_loss_aux = center_loss_func(out_aux, target)

            # print('se_connectivity_loss: ', se_connectivity_loss)
            # print('se_connectivity_loss_aux: ', se_connectivity_loss_aux)
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self.num_classes

            loss = self.loss(output, target)
            loss_aux = self.loss(out_aux, target)

            # print('loss: ', loss)
            # print('loss_aux', loss_aux)

            if isinstance(self.loss, torch.nn.DataParallel):
                print('normal')
                loss = loss.mean()
                loss_aux = loss_aux.meam()

            # if isinstance(se_connectivity_loss, torch.nn.DataParallel):
            #     print('se')
            #     se_connectivity_loss = se_connectivity_loss.mean()
            #     se_connectivity_loss_aux = se_connectivity_loss_aux.mean()

            # if isinstance(center_loss, torch.nn.DataParallel):
            #     print('center')
            #     center_loss = center_loss.mean()
            #     center_loss_aux = center_loss_aux.mean()

            # se_loss_plus = .3 * (se_connectivity_loss + theta * se_connectivity_loss_aux)
            # center_loss_plus = center_loss + theta * center_loss_aux
            loss_plus = loss + theta * loss_aux

            # loss1 = loss_plus + .1 * se_loss_plus + .6 * center_loss_plus
            loss1 = loss_plus
            loss1.backward()
            self.optimizer.step()
            self.total_loss.update(loss1.item())
            # print('loss1: ', loss1.item())
            # print('loss: ', loss1.item())
            # print('se_connectivity_loss: ', loss1.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, Class_IoU = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} | {} | {:.3f} |'.format(
                    epoch, self.total_loss.average,
                    pixAcc, mIoU,
                    self.batch_time.average, self.data_time.average, Class_IoU, loss_plus.item()))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
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
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):  # [0,1,0,1]
                # data, target = data.to(self.device), target.to(self.device)
                # LOSS
                out_aux, output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
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
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
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