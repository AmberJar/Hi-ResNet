import os
import logging
import json
import math
import sys
import dataloaders
import torch
import datetime
# from torch.utils import tensorboard
from torch.utils import tensorboard
from torch import nn
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import Logger
from utils.cyclical_learning_rates import ScheduledOptim
from timm.utils import ModelEmaV2
from utils.training_tools import EMA
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def get_instance_batch_size(module, name, batch_size, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    config[name]['args']['batch_size'] = batch_size
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader=None, val_loader=None, local_rank=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_logger = Logger()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE
        # self.device, available_gpus = self._get_available_devices(self.config['n_gpu'])
        _, available_gpus = self._get_available_devices(self.config['n_gpu'])
        # 定义distributed Data Parallel所需要的一些参数
        '''
        backend: 后端, 实际上是多个机器之间交换数据的协议
        init_method: 机器之间交换数据, 需要指定一个主节点, 而这个参数就是指定主节点的,推荐使用环境变量初始化，就是在你使用函数的时候不需要填写该参数即可，默认使用环境变量初始化
        world_size: 介绍都是说是进程, 实际就是机器的个数, 例如两台机器一起训练的话, world_size就设置为2
        rank: 区分主节点和从节点的, 主节点为0, 剩余的为了1-(N-1), N为要使用的机器的数量, 也就是world_size
        '''
        torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="nccl",
        #                                      init_method=None,
        #                                      timeout=datetime.timedelta(0, 1800),
        #                                      world_size=-1,
        #                                      rank=-1,
        #                                      store=None)
        self.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        # ======================================================================================================
        if config["path_best"]:
            print("Load Model Weights")  # Load checkpoint
            # root of checkpoint
            path_best = config["path_best"]
            checkpoint = torch.load(path_best, map_location=self.device)
            checkpoint = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
            model.load_state_dict(checkpoint, strict=False)
        # ======================================================================================================

        # 分布式训练的时候使用Use_synch_bn，用于平衡各个地方的bn的结果，达到一个更好的效果，在batch_size较小的时候使用
        # data distributed

        self.model.to(self.device)

        # EMA
        if config['ema']['use']:
            ema_weight = config['ema']['weight']
            self.ema = EMA(self.model, ema_weight)
            self.ema.register()
        else:
            self.ema = None

        if config["use_synch_bn"]:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank)

        # batch_size
        self.train_batch_size = self.config['train_loader']['args']['batch_size'] // torch.cuda.device_count()
        self.val_batch_size = self.config['val_loader']['args']['batch_size'] // torch.cuda.device_count()

        self.train_loader = get_instance_batch_size(dataloaders, 'train_loader', self.train_batch_size, self.config)
        self.val_loader = get_instance_batch_size(dataloaders, 'val_loader', self.val_batch_size, self.config)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.warm_up_epoch = self.config['lr_scheduler']['args']['warmup_epochs']

        # OPTIMIZER
        # 过滤正常训练的参数，让backbone少动一点，decode多动一点
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_decoder_params())},
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                     'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                                     'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)  # 根据字符串和json文件读取optimizer的参数

        if config['lr_scheduler']['type'] == 'CosineAnnealingLRWarmup':
            self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(self.optimizer,
                                                                                            T_max=100,
                                                                                            eta_min=1.0e-4,
                                                                                            last_epoch=-1,
                                                                                            warmup_steps=10,
                                                                                            warmup_start_lr=1.0e-5)
        else:
            self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(optimizer=self.optimizer,
                                                                                            num_epochs=self.epochs,
                                                                                            iters_per_epoch=len(self.train_loader),
                                                                                            warmup_epochs=self.warm_up_epoch)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time)
        if self.local_rank == 0:
            helpers.dir_exists(self.checkpoint_dir)
            config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
            with open(config_save_path, 'w') as handle:
                json.dump(self.config, handle, indent=4, sort_keys=True)

            writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time)
            writer_dir = writer_dir.replace('\\', '/')
            self.writer = tensorboard.SummaryWriter(writer_dir)

        if resume: self._resume_checkpoint(resume)


    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # self.train_loader.train_sampler.set_epoch(epoch)
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch)
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)

                # LOGGING INFO
                self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if (epoch % self.save_period == 0) and self.local_rank == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')

        checkpoint = torch.load(resume_path, map_location='cpu')
        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        # (checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
