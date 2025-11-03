import torch

def add_square_trigger(img_tensor, size=3, value=1.0, loc="br"):
    img = img_tensor.clone()
    _, h, w = img.shape
    if loc == "br":
        img[:, h-size:h, w-size:w] = value
    return img

def poison_dataset(images, labels, poison_rate=0.01, target_label=0):
    n = len(labels)
    k = int(poison_rate * n)
    idx = torch.randperm(n)[:k]
    images_poison = images.clone()
    labels_poison = labels.clone()
    for i in idx:
        images_poison[i] = add_square_trigger(images_poison[i])
        labels_poison[i] = target_label
    return images_poison, labels_poison, idx
