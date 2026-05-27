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

- [swin-transformer-cifar10.ipynb](./swin-transformer-cifar10.ipynb)
  - `CIFAR-10`，`224x224`
  - Window Attention（W-MSA）、Shifted Window Attention（SW-MSA）、Patch Merging、层次化特征图

### 序列建模

- [transformer-translation.ipynb](./transformer-translation.ipynb)
  - Multi30k 英德翻译数据集
  - Positional Encoding、Multi-head Self-Attention、Cross-Attention、Encoder-Decoder 结构、Greedy Decoding

- [gpt2-causal-lm.ipynb](./gpt2-causal-lm.ipynb)
  - TinyShakespeare，字符级语言模型
  - 可学习绝对位置 Embedding、Causal Self-Attention、GeLU MLP、Pre-LayerNorm

- [llama-causal-lm.ipynb](./llama-causal-lm.ipynb)
  - TinyShakespeare，字符级语言模型
  - RMSNorm、RoPE、**GQA（Grouped Query Attention）**、SwiGLU、Pre-Norm、温度采样

### 生成模型

- [ddpm-cifar10.ipynb](./ddpm-cifar10.ipynb)
  - `CIFAR-10`，`32x32`
  - 前向加噪过程、线性 Beta Schedule、U-Net 去噪网络、时间步嵌入、DDPM 采样算法

- [stable-diffusion.ipynb](./stable-diffusion.ipynb)
  - 文本提示图像生成
  - VAE Encoder/Decoder、CLIP 文本编码、Latent Diffusion U-Net、调度器（PNDM/DDIM）、Classifier-Free Guidance

### 目标检测

- [yolo-object-detection.ipynb](./yolo-object-detection.ipynb)
  - COCO 类别，自然图像输入
  - YOLOv8 推理流程、CSP Backbone、PANet FPN Neck、检测头、NMS 后处理、边界框可视化

- [detr-object-detection.ipynb](./detr-object-detection.ipynb)
  - COCO 类别，自然图像输入
  - ResNet-50 Backbone、Transformer Encoder/Decoder、Object Query、二分图匹配、Cross-Attention 可视化

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
- diffusers
- datasets
- ultralytics
- pillow

可使用以下命令安装基础依赖：

```bash
pip install torch torchvision matplotlib transformers diffusers datasets ultralytics pillow
```

## 阅读路径

1. 先看 `AlexNet -> VGG16 -> ResNet18 -> DenseNet121`，建立经典视觉模型演化脉络。
2. 再看 `ViT -> Swin Transformer`，对比 CNN 和 Transformer 的视觉建模差异，以及全局 vs 窗口注意力。
3. 看 `Transformer（翻译任务）`，理解 Encoder-Decoder 结构与序列生成原理。
4. 看 `GPT-2 -> LLaMA`，对比 Decoder-only 语言模型的演化（绝对 PE → RoPE，MHA → GQA，GeLU → SwiGLU）。
5. 看 `DDPM -> Stable Diffusion`，从基础扩散模型到潜在扩散生成流程。
6. 看 `YOLO -> DETR`，对比密集预测（Anchor-free）和集合预测（Transformer）两种检测范式。
7. 最后看 `CLIP`，切到图文共同嵌入和零样本分类。
8. 看 `SimCLR / MoCo / BYOL`，对比自监督表示学习三种典型路线。

## 对应论文

### 经典监督视觉模型

- AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- VGG / VGG16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- ResNet / ResNet18: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- DenseNet / DenseNet121: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- Vision Transformer (ViT): [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Swin Transformer: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

### 序列建模

- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 生成模型

- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Stable Diffusion / LDM: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### 目标检测

- YOLO: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

### 语言模型

- GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- LLaMA 2: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- GQA: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

### 多模态模型

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### 自监督表示学习

- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- MoCo: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
- BYOL: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)

## 说明

这里统一将 `CIFAR-10` 图像拉伸到 `224x224`，主要目的是让经典 CNN、ViT 和自监督模型在更接近标准视觉输入设定的条件下进行结构分析、特征分析和实现对比。
