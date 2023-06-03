import os
import json
import argparse
import sys

import torch
import dataloaders
import models
from utils import losses
from utils import Logger
from hrnet_trainer_s import Trainer
import os
from torch import nn
import random
import numpy as np
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# from thop import profile
# from thop import clever_format
from collections import OrderedDict
from torchstat import stat

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])    #getattar(类名，属性）（方法）


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def initialize_weights(*models):
    """
    初始化模型的weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # _calculate_correct_fan(tensor, mode)是算出input和output feature
                # map的元素总数
                # Fills the input Tensor with values according to the method
                # described in Delving deep into rectifiers: Surpassing human-
                # level performance on ImageNet classification - He, K. et al.
                # (2015), using a normal distribution. The resulting tensor will
                # have values sampled from \mathcal{N}(0, \text{std}^2)N(0,std2)
                # where 公式在底部

                nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Fills self tensor with the specified value.
                # 把weight初始化为1，
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def main(config, resume, local_rank, pretrian=True):
    seed_everything(88)
    train_logger = Logger()

    # DATA LOADERS
    # train_loader = get_instance(dataloaders, 'train_loader', config)
    # val_loader = get_instance(dataloaders, 'val_loader', config)

    # classes
    # num_classes = 66
    num_classes = 2
    # MODEL
    model = get_instance(models, 'arch', config, num_classes)

    if config['use_pretrained']:
        print('ALL SUCCESS')
        checkpoint = torch.load(config['pretrained_path'], map_location=torch.device('cpu'))

        # for k, v in checkpoint.items():
        #     print(k)

        # scenario1
        # state_dict = model.state_dict()
        # model_dict = {}
        #
        # for k, v in checkpoint['state_dict'].items():
        #     model_dict[k] = v
        #
        # state_dict.update(model_dict)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

        # scenario2
        # for cpu inference, remove module
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     # print(k)
        #     name = k[7:]
        #     new_state_dict[name] = v
        # checkpoint = new_state_dict

        # scenario3
        # del checkpoint['state_dict']['cls_head.weight']
        # del checkpoint['state_dict']['aux_head.2.weight']

        # del checkpoint['state_dict']['asp_ocr_head.context.0.weight']
        # del checkpoint['state_dict']['asp_ocr_head.conv2.0.weight']
        # del checkpoint['state_dict']['asp_ocr_head.conv3.0.weight']
        # del checkpoint['state_dict']['asp_ocr_head.conv4.0.weight']
        # del checkpoint['state_dict']['asp_ocr_head.conv5.0.weight']
        # del checkpoint['state_dict']['aux_head.0.weight']
        # del checkpoint['state_dict']['asp_ocr_head.conv4.0.weight:']

        # # state_dict = checkpoint['model'].state_dict()


        # from collections import OrderedDict
        del_list_1 = ['cls_head.weight', 'aux_head.2.weight']
        del_list_2 = ["fc1.0.weight", "fc1.0.bias", "fc1.2.weight", "fc1.2.bias",
                      "fc2.0.weight", "fc2.0.bias", "fc2.2.weight", "fc2.2.bias",
                      'aux_head.2.weight']
        imagenet = False
        if not imagenet:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.encoder_q.'):
                    # name = k[7:]  # 去掉 `module.`
                    name = k.replace("module.encoder_q.", "", 1)
                    if name in del_list_2:
                        continue
                    new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
        else:
            new_state_dict = OrderedDict()
            # print(checkpoint['cls_head.weight'])
            for k, v in checkpoint.items():
                name = k[7:]  # 去掉 `module.`
                # if name in del_list:
                #     continue
                new_state_dict[name] = v
            model.load_state_dict(checkpoint, strict=False)

        # model.load_state_dict(checkpoint['state_dict'], strict=True)

        model.eval()


    # initialize_weights(model)
    # model_structure(model)

    # 参数量计算
    # 1
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .2fM' % (total / 1e6))
    # sys.exit()
    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # memory and flops
    # print(stat(model, (3, 224, 224)))

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        local_rank=local_rank,
        train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    # parser = argparse.ArgumentParser(description='PyTorch Training').add_argument()
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config_loveda.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='2,1,3,0', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    config = json.load(open(args.config))
    data_dir = config['train_loader']['args']['data_dir']
    dataloader = config['train_loader']['type']
    n_gpu = config['n_gpu']
    batch_size = config["train_loader"]['args']['batch_size']

    if args.resume:
        config = torch.load(args.resume, map_location='cpu')['config']
        # config['train_loader']['type'] = dataloader
        config['train_loader']['args']['data_dir'] = data_dir
        # config['val_loader']['type'] = dataloader
        config['val_loader']['args']['data_dir'] = data_dir
        config["n_gpu"] = n_gpu

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.local_rank)
