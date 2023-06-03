# adaspace_aerial_image_to_2dmaps

#### 介绍
AI算法实现遥感影像数据到2D矢量化地图的生成，由2D矢量地图生成3D场景项目开发仓库

#### 软件架构
```
WorkingDirection/
├────base/
│    ├────__init__.py
│    ├────base_dataloader.py
│    ├────base_dataloader_val.py
│    ├────base_dataset.py
│    ├────base_dataset_val.py
│    ├────base_model.py
│    ├────base_trainer.py
│    ├────based_dataloader_val.py
│    ├────image_enhance.py
│    ├────image_noise.py
│    └────white_balance.py
├────ProcessingTools/
│    ├────calculate_mean_std.py - 计算数据集的mean和std
│    ├────change_labels.py - 修改label的值，用于合并分类
│    ├────change_name.py - 修改文件的名字
│    ├────data_augmentation.py - 数据增强
│    ├────fill_drop.py - 填孔
│    ├────generate_context.py - 生成文件树
│    ├────images_resize.py - 对图片进行上采样和下采样
│    ├────label_filter.py - 过滤label
│    ├────mask2source.py - 将mask印在图片上
│    ├────mask_in_pic.py - 未完成版本
│    ├────merge_mask.py - 将mask随机合并
│    ├────pic_concat.py - 图片拼接
│    ├────pic_imp.py - 图片补齐
│    ├────RGB2GRAY.py - RGB转灰度
│    ├────seperate_train_test.py - 将同一文件夹下的train和test分离
│    ├────split_image.py - 切割图片版本1
│    ├────split_image_V2.py - 切割图片版本2
│    ├────split_images.py - 批量切割图片
│    ├────split_train_val.py - 将数据集分为train和validation
│    └────white_balance.py - 白平衡
├────config.json    -   原始的config
├────config_hrnet_cpam_hdloss.json  -   hrnet的config
├────config_hrnet_test.json - 用于测试的hrnetconfig
├────config_hrnet_val.json  - 道路连接算法的config
├────config_uppernet.json   - unet++ 的config
├────dataloaders/   -   不同数据集的dataloader
│    ├────__init__.py
│    ├────adaspace.py
│    ├────ade20k.py
│    ├────cityscapes.py
│    ├────coco.py
│    ├────huawei.py - 一般使用的dataloader
│    ├────labels/
│    │    ├────adaspaceunify.txt
│    │    ├────ade20k.txt
│    │    ├────cityscapes.txt
│    │    ├────coco.txt
│    │    ├────cocostuff_hierarchy.json
│    │    ├────cocostuff_labels.txt
│    │    ├────TEXT.py
│    │    ├────voc.txt
│    │    ├────voc_context_classes-400.txt
│    │    └────voc_context_classes-59.txt
│    ├────road.py   -  道路预测算法的dataloader
│    ├────thunder.py    - 道路连接算法的dataloader
│    └────voc.py
├────hrnet_train.py - 训练水体、道路预测的train文件
├────hrnet_train_test.py - hrnet测试训练的train文件
├────hrnet_trainer_f.py - 训练水体、道路预测的trainer文件
├────hrnet_trainer_test.py - hrnet测试训练的trainer文件
├────hrnet_val_train.py - 道路连接算法的train文件
├────hrnet_val_trainer.py - 道路连接算法的trainer文件
├────image_enhance.py - 数据增强
├────inference.py - 一般推理
├────inference_val.py - 道路连接算法推理
├────inferenece_TTA.py - 使用TTA进行推理
├────models/ - 集成的模型
│    ├──── deeplabv3_plus_xception.py
│    ├────__init__.py
│    ├────deeplabv3_plus.py
│    ├────deeplabv3_plus_edge_cpam.py
│    ├────duc_hdc.py
│    ├────effnetv2.py
│    ├────effnetv2_unet.py
│    ├────enet.py
│    ├────fcn.py
│    ├────gcn.py
│    ├────hrnet.py - 一般hrnet模型
│    ├────hrnet_backbone.py
│    ├────hrnet_backboneV2.py
│    ├────hrnet_backboneV3.py
│    ├────hrnet_config.py
│    ├────hrnet_configV2.py
│    ├────hrnet_configV3.py
│    ├────hrnet_spatial_ocr_block.py
│    ├────hrnet_spatial_ocr_blockV2.py
│    ├────hrnet_spatial_ocr_blockV3.py
│    ├────hrnetV2.py - 道路连接模型
│    ├────hrnetV3.py - 网络修改测试模型
│    ├────pspnet.py
│    ├────resnet.py
│    ├────segnet.py
│    ├────unet.py
│    └────upernet.py
├────README.md
├────requirements.txt
├────split_images.py
├────uppernet_train.py - unet++的train文件
└────uppernet_trainer.py - unet++的trainer文件

  ```

#### 安装教程
pip install -r requirements.txt

#### 使用说明
1.  训练
```bash
python train.py --config config.json
```
2.  可视化
```bash
tensorboard --logdir saved
```
3.  推理
```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```
推理可选参数:
```
--output       The folder where the results will be saved (default: outputs).
--extension    The extension of the images to segment (default: jpg).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--mode         Mode to be used, choose either `multiscale` or `sliding` for inference (multiscale is the default behaviour).
--config       The config file used for training the model.
```

#### 模型
1. (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)
2. (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper]](https://arxiv.org/abs/1703.02719)
3. (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper]](https://arxiv.org/abs/1807.10221)
4. (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1702.08502) 
5. (**PSPNet**) Pyramid Scene Parsing Network [[Paper]](http://jiaya.me/papers/PSPNet_cvpr17.pdf) 
6. (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper]](https://arxiv.org/abs/1606.02147)
7. (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
8. (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper]](https://arxiv.org/pdf/1511.00561)
9. (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015): [[Paper]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

#### 配置文件
Config files are in `.json` format:
```javascript
{
  "name": "PSPNet",         // training session name
  "n_gpu": 1,               // number of GPUs to use for training.
  "use_synch_bn": true,     // Using Synchronized batchnorm (for multi-GPU usage)

    "arch": {
        "type": "PSPNet", // name of model architecture to train
        "args": {
            "backbone": "resnet50",     // encoder type type
            "freeze_bn": false,         // When fine tuning the model this can be used
            "freeze_backbone": false    // In this case only the decoder is trained
        }
    },

    "train_loader": {
        "type": "VOC",          // Selecting data loader
        "args":{
            "data_dir": "data/",  // dataset path
            "batch_size": 32,     // batch size
            "augment": true,      // Use data augmentation
            "crop_size": 380,     // Size of the random crop after rescaling
            "shuffle": true,
            "base_size": 400,     // The image is resized to base_size, then randomly croped
            "scale": true,        // Random rescaling between 0.5 and 2 before croping
            "flip": true,         // Random H-FLip
            "rotate": true,       // Random rotation between 10 and -10 degrees
            "blur": true,         // Adding a slight amount of blut to the image
            "split": "train_aug", // Split to use, depend of the dataset
            "num_workers": 8
        }
    },

    "val_loader": {     // Same for val, but no data augmentation, only a center crop
        "type": "VOC",
        "args":{
            "data_dir": "data/",
            "batch_size": 32,
            "crop_size": 480,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,      // Using lr/10 for the backbone, and lr for the rest
        "args":{
            "lr": 0.01,               // Learning rate
            "weight_decay": 1e-4,     // Weight decay
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",     // Loss (see utils/losses.py)
    "ignore_index": 255,              // Class to ignore (must be set to -1 for ADE20K) dataset
    "lr_scheduler": {   
        "type": "Poly",               // Learning rate scheduler (Poly or OneCycle)
        "args": {}
    },

    "trainer": {
        "epochs": 80,                 // Number of training epochs
        "save_dir": "saved/",         // Checkpoints are saved in save_dir/models/
        "save_period": 10,            // Saving chechpoint each 10 epochs
  
        "monitor": "max Mean_IoU",    // Mode and metric for model performance 
        "early_stop": 10,             // Number of epochs to wait before early stoping (0 to disable)
        
        "tensorboard": true,        // Enable tensorboard visualization
        "log_dir": "saved/runs",
        "log_per_iter": 20,         

        "val": true,
        "val_per_epochs": 5         // Run validation each 5 epochs
    }
}      
#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技
test

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)


#### 使用说明

1. 一般hrnet的注释写在hrnet_train以及hrnet_trainer_f中
2. 道路连接hrnet的注释写在hrnet_val_train以及hrnet_val_trainer中
3. 一模训练使用hrnet即可，修改config中的模型以切换不同的模型，无须使用其他的train
4. 多个train只是为了方便训练的时候可以不需要修改参数
5. 训练前需要修改dataloader中的num_classes以及glob的文件后缀

