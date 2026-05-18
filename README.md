# Foundation Model Analysis

这个目录用于整理视觉基础模型、视觉 Transformer、多模态模型和自监督表示学习方法的结构解读与实验 Notebook。

内容重点包括：

- 数据集加载与预处理流程
- 模型结构拆解
- 特征图或 token 变化分析
- 训练、验证与预测代码
- feature 提取与表示学习流程
- 模型设计逻辑与论文对应关系

## 内容总览

### 经典监督视觉模型

- [alexnet-cifar10.ipynb](./alexnet-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - 经典 CNN 起点，大卷积核、池化层、全连接分类头

- [vgg16-cifar10.ipynb](./vgg16-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - 深层纯卷积堆叠，连续 `3x3` 卷积结构

- [resnet18-cifar10.ipynb](./resnet18-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - 残差连接、阶段式下采样、ResNet 基本结构

- [densenet121-cifar10.ipynb](./densenet121-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - Dense Block、Transition Layer、特征复用与通道拼接

- [vit-cifar10.ipynb](./vit-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - Patch Embedding、Class Token、Position Embedding、Transformer Encoder

### 多模态模型

- [clip-vit-l14-feature-classification.ipynb](./clip-vit-l14-feature-classification.ipynb)
  - 单图输入，自定义类别文本
  - `CLIP ViT-L/14`，图像特征提取、文本特征提取、零样本分类

### 自监督表示学习

- [simclr-cifar10.ipynb](./simclr-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - 双视图增强、projection head、`NT-Xent / InfoNCE`、线性评估

- [moco-cifar10.ipynb](./moco-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - query/key encoder、momentum update、queue memory bank、InfoNCE

- [byol-cifar10.ipynb](./byol-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - online/target network、predictor、BYOL loss、无负样本训练

## 运行环境

- Python 3
- PyTorch
- torchvision
- matplotlib
- transformers
- pillow

可使用以下命令安装基础依赖：

```bash
pip install torch torchvision matplotlib transformers pillow
```

## 阅读路径

1. 先看 `AlexNet -> VGG16 -> ResNet18 -> DenseNet121`，建立经典视觉模型演化脉络。
2. 再看 `ViT`，对比 CNN 和 Transformer 的视觉建模差异。
3. 再看 `CLIP`，切到图文共同嵌入和零样本分类。
4. 最后看 `SimCLR / MoCo / BYOL`，对比自监督表示学习三种典型路线。

## 对应论文

### 经典监督视觉模型

- AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- VGG / VGG16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- ResNet / ResNet18: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- DenseNet / DenseNet121: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- Vision Transformer (ViT): [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### 多模态模型

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### 自监督表示学习

- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- MoCo: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
- BYOL: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)

## 说明

这里统一将 `CIFAR-10` 图像拉伸到 `224x224`，主要目的是让经典 CNN、ViT 和自监督模型在更接近标准视觉输入设定的条件下进行结构分析、特征分析和实现对比。
