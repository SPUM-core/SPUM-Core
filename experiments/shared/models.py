"""
共享模型定义 — P0/P1/P2/N015/N016 统一使用

模型:
    MLP(features, hidden_dim, num_classes)
        — 2层全连接，ReLU 激活，支持 return_hidden
    DeepCNN(num_classes)
        — 8层卷积 + 1层全连接 (N016 架构: 287,722 参数)

所有模型输出层不包含 Softmax — 交叉熵损失内部处理。
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """2层全连接 MLP，支持返回隐藏层表示。

    用法:
        model = MLP(features=784, hidden_dim=128, num_classes=10)
        logits, hidden = model(x, return_hidden=True)  # P0/P2 需要 hidden

    自适应:
        - features=20, hidden_dim=64  → 合成数据
        - features=784, hidden_dim=64 → MNIST (轻量)
        - features=784, hidden_dim=128 → MNIST (标准)
    """
    def __init__(self, features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return (out, h) if return_hidden else out


class DeepCNN(nn.Module):
    """N016 DeepCNN 架构: 8层卷积 + 1层全连接。

    参数: ~287,722 (MNIST 28x28)
    无 BatchNorm (保持架构与 N016 实验一致)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(128, num_classes)
        self.h = None

    def forward(self, x):
        x = self.conv(x)
        self.h = x.view(x.size(0), -1)  # 暴露隐藏层供 SPUM 分析
        return self.out(self.h)
