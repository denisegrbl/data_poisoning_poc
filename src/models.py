# src/models.py
import torch
import torch.nn as nn
import torchvision.models as tvm

# ---------- MNIST ----------
class SimpleCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        # Feature-Teil
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),           # 14->7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU()
        )
        # Klassifikationskopf
        self.head = nn.Linear(128, num_classes)

    def forward_features(self, x):
        return self.feat(x)

    def forward(self, x):
        f = self.forward_features(x)
        return self.head(f)

# ---------- CIFAR: kleines CNN ----------
class SmallCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU()
        )
        self.head = nn.Linear(256, num_classes)

    def forward_features(self, x):
        return self.feat(x)

    def forward(self, x):
        f = self.forward_features(x)
        return self.head(f)

# ---------- CIFAR: ResNet18 Wrapper (mit forward_features) ----------
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        m = tvm.resnet18(weights=None)
        # CIFAR-angepasste erste Schicht
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(m.fc.in_features, num_classes)

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # (N, 512)
        return x

    def forward(self, x):
        f = self.forward_features(x)
        return self.fc(f)

# ---------- Fabrikfunktion ----------
def get_model(dataset: str, arch: str, num_classes: int, in_channels: int):
    """
    dataset: 'mnist' oder 'cifar10'
    arch:    'auto' | 'simplecnn' | 'resnet18' (bei MNIST ignoriert)
    """
    if arch == "auto":
        arch = "simplecnn" if dataset == "mnist" else "resnet18"

    if dataset == "mnist":
        return SimpleCNN_MNIST(num_classes=num_classes, in_channels=in_channels)

    # CIFAR10
    if arch == "resnet18":
        return ResNet18_CIFAR(num_classes=num_classes, in_channels=in_channels)
    else:
        return SmallCNN_CIFAR(num_classes=num_classes, in_channels=in_channels)
