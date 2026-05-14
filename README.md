# STL10 图像分类项目

基于 PyTorch 的 STL10 数据集图像分类项目，支持多种 CNN 模型架构、数据增强策略和模型可解释性分析。

## 项目结构

```
.
├── src/
│   ├── train.py              # 模型训练脚本
│   ├── eval.py               # 模型评估脚本
│   ├── dataset/
│   │   └── dataset.py        # STL10 数据集加载器
│   ├── model/
│   │   ├── base.py           # 基础 CNN 模型
│   │   ├── deeper.py         # 加深版 CNN 模型
│   │   ├── res.py            # 残差网络模型
│   │   └── SEBlock.py        # SE / CBAM 注意力模型
│   └── tool/
│       ├── metric.py         # 分类评估指标
│       ├── gradcam.py        # Grad-CAM 可视化
│       ├── feature_vis.py    # PCA/t-SNE 特征可视化
│       └── visualize.py      # 训练曲线可视化
├── out/                      # 训练输出目录
│   ├── train/                # 训练日志和模型权重
│   ├── test/                 # 测试评估日志
│   └── gradcam/              # Grad-CAM 可视化结果
├── STL10/                    # STL10 数据集（需自行下载）
│   ├── train/
│   └── test/
└── README.md
```

## 环境配置

### 依赖环境

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- scikit-learn
- Pillow

### 安装命令

```bash
pip install torch torchvision numpy matplotlib scikit-learn Pillow
```

<br />

## 快速开始

### 1. 模型训练

```bash
cd src
python train.py
```

**主要超参数**（在 `train.py` 中修改）：

```python
BATCH_SIZE = 32        # 批次大小
EPOCHS = 200           # 训练轮数
LR = 0.01              # 学习率
RESUME = True          # 是否从检查点恢复训练
```

**数据增强配置**：

```python
TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # 随机水平翻转
    transforms.RandomRotation(8),          # 随机旋转 ±8°
    transforms.ColorJitter(                 # 颜色抖动
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.1
    ),
    transforms.ToTensor(),
])
```

### 2. 模型评估

```bash
cd src
python eval.py
```

评估脚本会自动：

- 计算每个类别的 Precision、Recall、F1-Score
- 计算平均指标（macro avg）
- 输出混淆分析

### 3. 模型切换

在 `train.py` 中通过注释切换不同模型：

```python
# 基础 CNN
from model.base import CNNModel
model = CNNModel().to('cuda')

# 残差网络（默认推荐）
from model.res import CNNResModel
model = CNNResModel().to('cuda')

# SE 注意力网络
from model.SEBlock import CNN_SE
model = CNN_SE().to('cuda')

# CBAM 注意力网络
from model.SEBlock import CNN_CBAM
model = CNN_CBAM().to('cuda')

# 加深版 CNN
from model.deeper import CNNModel
model = CNNModel().to('cuda')
```

## 模型架构

### 1. 基础 CNN (CNNModel)

简单的 4 层卷积网络：

- Conv(3→32) → BN → Pool → Dropout(0.2)
- Conv(32→64) → BN → Pool → Dropout(0.2)
- Conv(64→128) → BN → Pool → Dropout(0.2)
- Conv(128→256) → BN → Pool → Dropout(0.2)
- AdaptiveAvgPool → Flatten → FC(256→128) → FC(128→10)

### 2. 残差网络 (CNNResModel)

带残差连接的 4 层网络：

- ResidualBlock(3→32) → Pool → Dropout(0.2)
- ResidualBlock(32→64) → Pool → Dropout(0.2)
- ResidualBlock(64→128) → Pool → Dropout(0.2)
- ResidualBlock(128→256) → Pool → Dropout(0.2)
- AdaptiveAvgPool → Flatten → FC(256→128) → FC(128→10)

### 3. SE 注意力网络 (CNN\_SE)

带通道注意力（Squeeze-and-Excitation）：

- Conv + BN + SEBlock → Pool
- Conv + BN + SEBlock → Pool
- Conv + BN + SEBlock → Pool
- Conv + BN + SEBlock → Pool
- AdaptiveAvgPool → FC(256→128) → FC(128→10)

### 4. CBAM 注意力网络 (CNN\_CBAM)

带通道+空间注意力（Convolutional Block Attention Module）：

- Conv + BN + CBAM → Pool
- Conv + BN + CBAM → Pool
- Conv + BN + CBAM → Pool
- Conv + BN + CBAM → Pool
- AdaptiveAvgPool → FC(256→128) → FC(128→10)

### 5. 加深版 CNN (deeper.CNNModel)

6 层卷积网络（512 通道）：

- Conv(3→32) → BN → Pool → Dropout
- Conv(32→64) → BN → Pool → Dropout
- Conv(64→128) → BN → Pool → Dropout
- Conv(128→256) → BN → Pool → Dropout
- Conv(256→512) → BN → Pool → Dropout
- Conv(512→512) → BN → Pool → Dropout
- AdaptiveAvgPool → Flatten → FC(512→256) → FC(256→10)

## 可视化工具

### Grad-CAM 类激活可视化

生成模型决策热力图：

```bash
cd src/tool
python gradcam.py
```

输出：

- `out/gradcam/gradcam_*.png` - 每个类别的 Grad-CAM 可视化

**配置参数**：

```python
MODEL_TYPE = 'res'      # 模型类型
WEIGHT_PATH = '...'     # 模型权重路径
IMAGE_DIR = '...'       # 测试图片目录
OUTPUT_DIR = '...'      # 输出目录
num_samples = 3         # 每类采样数量
```

<br />

### 训练曲线可视化

绘制 loss 和 accuracy 曲线：

```bash
cd src/tool
python visualize.py
```

输出：

- `out/train/*/training_curves.png` - 训练曲线图

## 训练输出

训练完成后，`out/train/{experiment_name}/` 目录下会生成：

```
out/train/{experiment_name}/
├── checkpoint.pth          # 最新检查点（含模型、优化器、调度器状态）
├── best_model.pth         # 最佳模型权重（部分实验）
├── epoch_losses.npy       # 每轮平均 loss
├── epoch_accuracies.npy   # 每轮验证准确率
├── train.log             # 训练日志
└── training_curves.png    # 训练曲线
```

### 恢复训练

设置 `RESUME = True` 即可从检查点恢复：

```python
RESUME = True  # 开启恢复训练
```

