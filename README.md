# DenseNet-Tensorflow
Reimplementation of DenseNet on Image Recognition

[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)(DenseNet), won the Best Paper Award on CVPR 2017.

This is an (re-)implementation of DenseNet in TensorFlow for image recognition tasks. The (re-)implementation is based on official [Torch DenseNet](https://github.com/liuzhuang13/DenseNet) with [Tensorflow Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

![DenseNet Table](https://github.com/NanqingD/DenseNet-Tensorflow/blob/master/images/DenseNet_table.png)

In the paper, DenseNet-264 seems to be a typo, since there is no way the number of layers to be an even number. See DenseNet-121, 169, 201, if you add up, which is 1 + 6 x 2 + 1 + 12 x 2 + 1 + 64 x 2 + 1 + 48 x 2 + 1 + 1 = 265.

## Features
- [x] DenseNet-B, DenseNet-C, DenseNet-BC
- [x] DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-265
- [ ] Training on CIFAR
- [ ] Training on SVHN
- [ ] Training on ImageNet

## Requirement
### Tensorflow 1.4
```
python 3.5
tensorflow 1.4
CUDA  8.0
cuDNN 6.0
```

### Tensorflow 1.2
```
python 3.5
tensorflow 1.2
CUDA  8.0
cuDNN 5.1
```

### Installation
```
pip3 install -r requirements.txt
```

## Performance
### CIFAR10
| L=100, k=12 | Accuracy | Top 1    |
| ----------- |:--------:|:--------:|
| paper       | 94.08%   | 5.92     |
| repo        | 95.75%   | 4.25     |

![cifar10_train](https://github.com/NanqingD/DenseNet-Tensorflow/blob/master/images/train_cifar10_L100_k12.png)

![cifar10_val](https://github.com/NanqingD/DenseNet-Tensorflow/blob/master/images/val_cifar10_L100_k12.png)

### CIFAR100
| L=100, k=12 | Accuracy | Top 1    |
| ----------- |:--------:|:--------:|
| paper       | 75.85%   | 24.15    |
| repo        |          |          |