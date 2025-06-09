# ResNet表情符号分类器

这是一个基于ResNet架构的表情符号分类器，使用PyTorch实现。该模型能够对输入的表情符号图像进行分类。

## 环境要求

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- tensorboard

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd resnet-emoji-classifier
```

2. 创建并激活虚拟环境：
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. 安装依赖：
```bash
pip install torch torchvision pillow tensorboard
```

## 使用方法

### 训练模型

1. 数据集：
   - 在`data`目录下创建子文件夹，每个子文件夹对应一个表情类别
   - 将对应的表情图片放入相应的类别文件夹中

2. 运行训练脚本：
```bash
python train.py
```

### 测试模型

使用测试脚本进行预测：
```bash
python test.py
```


### 查看训练过程

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=logs
```

## 项目结构

```
resnet-emoji-classifier/
├── data/               # 数据集目录
├── logs/              # TensorBoard日志
├── model.pth          # 训练好的模型
├── train.py           # 训练脚本
├── test.py            # 测试脚本
└── check_env.py       # 环境检查脚本
```

## 项目说明

### 模型训练

 - 训练脚本采用ResNet残差网络结构，同时提供了一个轻量级CNN结构（见train_CNN.py），进行对比实验。
 - 使用交叉熵损失函数（CrossEntropyLoss）和Adam优化器。每轮训练后在验证集上评估模型性能，自动保存损失最小的模型权重。
 - 支持Early Stopping机制：若验证损失连续若干轮未提升，则提前终止训练，防止过拟合。
 - 训练和验证过程中的损失与准确率会记录到TensorBoard日志，便于可视化分析。

### 模型测试

测试脚本会自动加载`owndata`目录下的所有`.jpg`图片，对每张图片进行分类预测，
并根据预测结果将图片重命名为`class_类别编号_原文件名`的格式。
请确保待测试图片已放置在`owndata`目录下，且模型权重文件`model.pth`已存在于项目根目录。

## 注意事项

所有图片为128*128的jpg文件，下载后需按9：1划分为训练集的验证集。