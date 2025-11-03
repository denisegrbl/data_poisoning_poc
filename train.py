import torch, torch.nn as nn, torch.optim as optim
from src.models import SimpleCNN
from src.data_utils import get_mnist_loaders
from src.poisoning import poison_dataset, add_square_trigger

def to_loader(images, labels, batch_size=128, shuffle=True):
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def evaluate(model, test_imgs, test_lbls, device="cpu"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_imgs), 512):
            x = test_imgs[i:i+512].to(device)
            y = test_lbls[i:i+512].to(device)
            out = model(x).argmax(1)
            correct += (out==y).sum().item()
    return correct/len(test_lbls)

def eval_asr(model, test_imgs, target_label=0, device="cpu"):
    timgs = torch.stack([add_square_trigger(img) for img in test_imgs])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(timgs), 512):
            x = timgs[i:i+512].to(device)
            out = model(x).argmax(1).cpu()
            preds.append(out)
    preds = torch.cat(preds)
    return (preds == target_label).float().mean().item()

def train_model(model, loader, epochs=5, device="cpu", lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (trX, trY), (teX, teY) = get_mnist_loaders()

    poison_rate = 0.01
    target_label = 0
    trXp, trYp, poison_idx = poison_dataset(trX, trY, poison_rate, target_label)

    model = SimpleCNN()
    loader = to_loader(trXp, trYp, 128, True)
    train_model(model, loader, epochs=5, device=device)
    clean_acc = evaluate(model, teX, teY, device)
    asr = eval_asr(model, teX, target_label, device)
    print(f"[BASELINE] CleanAcc={clean_acc:.4f}  ASR={asr:.4f}")

if __name__ == "__main__":
    main()
