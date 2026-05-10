import torch
import torch.nn as nn
import torch.nn.functional as F

# 残差
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

# 加入残差的 CNN 模型
class CNNResModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = ResidualBlock(3, 32, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)

       
        self.res1 = ResidualBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)

        self.res2 = ResidualBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.2)

        self.res3 = ResidualBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout5 = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 初始卷积层
        x = self.pool1(F.relu(self.bn1(self.res(x))))
        x = self.dropout1(x)

        # 残差块 + 池化
        x = self.pool2(self.res1(x))
        x = self.dropout2(x)

        x = self.pool3(self.res2(x))
        x = self.dropout3(x)

        x = self.pool4(self.res3(x))
        x = self.dropout4(x)

        # 全局池化 + 分类
        x = self.gap(x)
        x = self.dropout5(x)
        x = self.classifier(x)
        return x