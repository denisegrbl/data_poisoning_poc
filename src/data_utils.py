# src/data_utils.py
import torch
import torchvision
import torchvision.transforms as T

def get_dataset_tensors(name="cifar10"):
    if name.lower() == "mnist":
        return _get_mnist()
    else:
        return _get_cifar10()

def _get_mnist():
    tfm = T.Compose([
        T.ToTensor(),  # [0,1], (1,28,28)
    ])
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    trX = torch.stack([train[i][0] for i in range(len(train))])
    trY = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    teX = torch.stack([test[i][0] for i in range(len(test))])
    teY = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)

    n_classes = 10
    in_channels = 1
    img_size = (28, 28)
    return (trX, trY), (teX, teY), n_classes, in_channels, img_size

def _get_cifar10():
    # Standard CIFAR-10 Normalisierung (optional – für PoC geht auch nur ToTensor)
    tfm = T.Compose([
        T.ToTensor(),  # (3,32,32)
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    trX = torch.stack([train[i][0] for i in range(len(train))])
    trY = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    teX = torch.stack([test[i][0] for i in range(len(test))])
    teY = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)

    n_classes = 10
    in_channels = 3
    img_size = (32, 32)
    return (trX, trY), (teX, teY), n_classes, in_channels, img_size
