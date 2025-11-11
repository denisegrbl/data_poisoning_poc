# src/data_utils.py
import os
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from .poisoning import add_square_trigger

# CIFAR-10 Normalisierung
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def _mnist_tensors(root="data"):
    tf = T.ToTensor()
    train_ds = torchvision.datasets.MNIST(root=root, train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf)
    trX = torch.stack([train_ds[i][0] for i in range(len(train_ds))])  # (N,1,28,28)
    trY = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
    teX = torch.stack([test_ds[i][0] for i in range(len(test_ds))])    # (N,1,28,28)
    teY = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
    return (trX, trY), (teX, teY), 10, 1, 28

def _cifar10_tensors(root="data", normalize=True):
    tf = T.Compose([T.ToTensor(), T.Normalize(_CIFAR_MEAN, _CIFAR_STD)]) if normalize else T.ToTensor()
    train_ds = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
    trX = torch.stack([train_ds[i][0] for i in range(len(train_ds))])  # (50000,3,32,32)
    trY = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
    teX = torch.stack([test_ds[i][0] for i in range(len(test_ds))])    # (10000,3,32,32)
    teY = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
    return (trX, trY), (teX, teY), 10, 3, 32

def get_dataset_tensors(name: str, root="data"):
    name = name.lower()
    if name == "mnist":
        return _mnist_tensors(root=root)
    elif name == "cifar10":
        return _cifar10_tensors(root=root, normalize=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def to_loader(images: torch.Tensor, labels: torch.Tensor, batch_size: int = 128, shuffle: bool = True):
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

# ---------- Visualisierung (Undo-Normalize für CIFAR-10) ----------

_CIFAR_MEAN_T = torch.tensor(_CIFAR_MEAN).view(3,1,1)
_CIFAR_STD_T  = torch.tensor(_CIFAR_STD).view(3,1,1)

def _unnorm_if_cifar(t: torch.Tensor, dataset: str):
    """t: (N,C,H,W) oder (C,H,W)"""
    if dataset.lower() != "cifar10":
        return t
    if t.dim() == 4:
        return (t * _CIFAR_STD_T + _CIFAR_MEAN_T).clamp(0,1)
    else:
        return (t * _CIFAR_STD_T + _CIFAR_MEAN_T).clamp(0,1)

def save_grid(tensor4d: torch.Tensor, path: str, nrow=4, dataset: str = "cifar10"):
    x = _unnorm_if_cifar(tensor4d.detach().cpu(), dataset)
    vutils.save_image(x, path, nrow=nrow, padding=2)

def save_sample_images(
    teX: torch.Tensor,
    out_dir: str,
    dataset: str,
    sample_idx: int = 1234,
    trigger_size: int = 3,
    trigger_value: float = 1.0,
    trigger_loc: str = "br",
    n_clean: int = 16,
):
    """
    Speichert konsistente Artefakte (immer gleicher sample_idx):
      - clean_grid.png (erste n_clean Testbilder)
      - poisoned_grid.png (dieselben n_clean mit Trigger)
      - clean_example_idx<idx>.png
      - poisoned_example_idx<idx>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # 16er-Grids
    grid_clean = teX[:n_clean]
    grid_poison = torch.stack([add_square_trigger(x, size=trigger_size, value=trigger_value, loc=trigger_loc)
                               for x in teX[:n_clean]])
    save_grid(grid_clean,  os.path.join(out_dir, "clean_grid.png"),    nrow=int(n_clean**0.5), dataset=dataset)
    save_grid(grid_poison, os.path.join(out_dir, "poisoned_grid.png"), nrow=int(n_clean**0.5), dataset=dataset)

    # Fixes Einzelbeispiel
    idx = min(max(sample_idx, 0), len(teX)-1)
    clean_one  = teX[idx:idx+1]                                  # (1,C,H,W)
    poison_one = torch.stack([add_square_trigger(clean_one[0],
                          size=trigger_size, value=trigger_value, loc=trigger_loc)])
    save_grid(clean_one,  os.path.join(out_dir, f"clean_example_idx{idx}.png"),    nrow=1, dataset=dataset)
    save_grid(poison_one, os.path.join(out_dir, f"poisoned_example_idx{idx}.png"), nrow=1, dataset=dataset)
