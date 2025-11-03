# src/poisoning.py
import os
import torch
import random
import numpy as np
from torchvision.utils import save_image

def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_square_trigger(img_tensor, size=3, value=1.0, loc="br"):
    """
    img_tensor: (C,H,W), Wertebereich [0,1]
    value: 0..1 (weiß=1.0)
    loc: br=bottom-right, tl, tr, bl
    """
    img = img_tensor.clone()
    _, h, w = img.shape
    if loc == "br":
        img[:, h-size:h, w-size:w] = value
    elif loc == "tr":
        img[:, 0:size, w-size:w] = value
    elif loc == "tl":
        img[:, 0:size, 0:size] = value
    elif loc == "bl":
        img[:, h-size:h, 0:size] = value
    return img

def poison_dataset_tensor(images, labels, poison_rate=0.01, target_label=0,
                          trigger_size=3, trigger_value=1.0, trigger_loc="br"):
    """
    Wählt k Bilder zufällig, fügt Trigger ein und setzt Label auf target_label.
    """
    n = len(labels)
    k = int(round(poison_rate * n))
    idx = torch.randperm(n)[:k]

    images_p = images.clone()
    labels_p = labels.clone()

    for i in idx:
        images_p[i] = add_square_trigger(images_p[i], size=trigger_size,
                                         value=trigger_value, loc=trigger_loc)
        labels_p[i] = target_label
    return images_p, labels_p, idx

def save_sample_images(test_imgs, target_label, out_dir, trigger_size=3, trigger_value=1.0, trigger_loc="br", n=16):
    os.makedirs(out_dir, exist_ok=True)
    # Clean Collage
    clean = test_imgs[:n].clone()
    save_image(clean, os.path.join(out_dir, "clean_grid.png"), nrow=4)
    # Poisoned Collage
    po = clean.clone()
    for i in range(len(po)):
        po[i] = add_square_trigger(po[i], size=trigger_size, value=trigger_value, loc=trigger_loc)
    save_image(po, os.path.join(out_dir, "poisoned_grid.png"), nrow=4)
