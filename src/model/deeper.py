import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 第1组
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)

        # 第2组
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)

        # 第3组
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.2)

        # 第4组
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.2)

        # ==================== 加深部分 ====================
        # 第5组（新增）
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2)
        self.dropout5 = nn.Dropout(0.2)

        # 第6组（新增，加深特征）
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2)  # 小池化，不压缩太多
        self.dropout6 = nn.Dropout(0.2)
        # ==================================================

        # 全局池化 + 分类头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout7 = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),  # 对应加深后的通道
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # 前向传播保持你原来的风格
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # 新增的两层卷积
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout5(x)
        
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout6(x)

        # 全局池化 + 全连接
        x = self.gap(x)
        x = self.dropout7(x)
        x = self.classifier(x)
        return x