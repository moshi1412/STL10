import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        # self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2)
        self.dropout2 = nn.Dropout(0.2)
        # self.gelu2 = nn.GELU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(2)
        self.dropout3 = nn.Dropout(0.2)
        # self.gelu3 = nn.GELU()
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AvgPool2d(2)
        self.dropout4 = nn.Dropout(0.2)
        # self.gelu4 = nn.GELU()


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout5 = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        x = self.gap(x)
        x = self.dropout5(x)
        x = self.classifier(x)
        return x