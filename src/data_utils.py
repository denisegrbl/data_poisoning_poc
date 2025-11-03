import torch
from torchvision import datasets, transforms

def get_mnist_loaders():
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    train_imgs = train.data.unsqueeze(1).float()/255.0
    train_lbls = train.targets.clone()
    test_imgs  = test.data.unsqueeze(1).float()/255.0
    test_lbls  = test.targets.clone()

    return (train_imgs, train_lbls), (test_imgs, test_lbls)
