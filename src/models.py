# src/models.py
import torch
import torch.nn as nn
import torchvision.models as tvm

class SimpleCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),           # 7x7
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class SmallCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

def get_model(dataset: str, arch: str, num_classes: int, in_channels: int):
    if arch == "auto":
        arch = "simplecnn" if dataset == "mnist" else "resnet18"

    if dataset == "mnist":
        return SimpleCNN_MNIST(num_classes=num_classes, in_channels=in_channels)

    # CIFAR10
    if arch == "resnet18":
        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        return SmallCNN_CIFAR(num_classes=num_classes, in_channels=in_channels)
