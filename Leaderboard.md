# Leaderboard

|                           模型名                            |  acc (%)  | 训练时间 |                    权重及训练日志下载地址                    |                             备注                             |
| :---------------------------------------------------------: | :-------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         ResNet50-E5                         |   86.01   | 6:50:28  | [百度网盘](https://pan.baidu.com/s/1qwW-jEnL5SnKull2Pl35Zw?pwd=l2ok)，[谷歌网盘](https://drive.google.com/file/d/1DdFcILRpM9jJR7a584_MYsUtF-DfH8QQ/view?usp=drive_link) |                                                              |
|                ResNet50-E5+Stochastic Depth                 |   86.12   | 6:53:32  |                                                              |                            dp=0.1                            |
|                ResNet50-E5+Stochastic Depth                 |   86.06   | 7:57:22  |                                                              |                            dp=0.2                            |
|                   ResNet50-E5+Drop Block                    |   85.82   | 8:12:54  |                                                              |                            db=0.1                            |
|                         WRN-50-2-E5                         |   85.74   | 12:35:44 |                                                              |                                                              |
|                        ResNet-101-E5                        |   85.50   | 10:02:03 |                                                              |                                                              |
|                        ResNet-152-E5                        |   86.38   | 14:57:37 |                                                              |                                                              |
|               ResNet-152-E5+Stochastic Depth                | **86.53** | 18:46:47 |                                                              | dp=0.5（跑实验的时候，还有其他的实验也占了大量显存，整张卡显存几乎占满了。训练时间仅供参考） |
|                      ResNet50-E6-tiny                       |   85.49   | 10:16:27 |                                                              |                                                              |
|                      ResNet50-E6-small                      |   86.32   | 19:47:55 |                                                              |                                                              |
|                WRN-18-10-E6 (resnet18_10_E6)                |   85.03   | 10:45:00 |                                                              |                                                              |
|             WRN-18-10-E6-v2 (resnet18_10_E6_v2)             |   85.22   | 7:48:08  |                                                              |                                                              |
|             WRN-26-10-E6-v2 (resnet26_10_E6_v2)             |   85.00   | 16:04:39 |                                                              |                                                              |
|             WRN-34-10-E6-v2 (resnet34_10_E6_v2)             |   85.55   | 19:06:49 |                                                              |                                                              |
|              ResNet272-E6-v2 (resnet272_E6_v2)              |   86.39   | 18:14:55 |                                                              |                         双卡RTX3090                          |
|           下面是WRN原文中定义的模型（未修改架构）           |           |          |                                                              |                                                              |
|                          WRN-16-8                           |   81.27   | 3:16:37  |                                                              |                                                              |
|                          WRN-28-10                          |   85.29   | 11:45:23 |                                                              |                                                              |
|                          WRN-40-10                          |   85.65   | 16:37:49 |                                                              |                                                              |
| 下面是ResNet原文中定义的处理CIFAR数据集的模型（未修改架构） |           |          |                                                              |                                                              |
|                          ResNet20                           |   66.87   | 1:31:27  |                                                              |                                                              |
|                          ResNet32                           |   71.18   | 2:08:42  |                                                              |                                                              |
|                          ResNet44                           |   73.56   | 2:48:14  |                                                              |                                                              |
|                          ResNet56                           |   75.27   | 3:24:39  |                                                              |                                                              |
|                          ResNet110                          |   77.90   | 6:23:43  |                                                              |                                                              |

上述结果默认训练脚本：

```bash
#!/bin/bash

for model in 'model_name'
do
    torchrun --nproc_per_node=1  --master_port="29429" classification/train.py \
        --model ${model} \
        --model_lib custom \
        --data_name cifar100 \
        --batch-size 128 \
        --lr 0.1 \
        --lr-scheduler cosineannealinglr \
        --epochs 300 \
        --lr-warmup-epochs 20 \
        --lr-min 1e-6 \
        --wd 5e-4 \
        --auto_augment \
        --random_erase 0.25 \
        --mixup-alpha 1 \
        --cutmix-alpha 1 \
        --act_layer relu \
        --loss_type ce \
        --print-freq 100 \
        --output-dir ./work_dir/aa-re_0.25-mixup-cutmix \
        --data-path /path/to/cifar100

    wait
done
```