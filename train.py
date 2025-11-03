# train.py
import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_utils import get_dataset_tensors
from src.models import get_model
from src.poisoning import (
    add_square_trigger,
    poison_dataset_tensor,
    save_sample_images,
    set_global_seed,
)

# -------------------------
# Argumente
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--arch", type=str, default="auto", choices=["auto", "simplecnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    # Poisoning / Backdoor
    p.add_argument("--poison_rate", type=float, default=0.0)  # 0.0 → Baseline
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--trigger_size", type=int, default=3)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--trigger_loc", type=str, default="br", choices=["br", "tl", "tr", "bl"])

    # Logging / Output
    p.add_argument("--exp_name", type=str, default="exp")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_samples", action="store_true")
    return p.parse_args()


# -------------------------
# Utility
# -------------------------
def to_loader(images, labels, batch_size=128, shuffle=True):
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def log_results(args, clean_acc, asr):
    ensure_dirs(args.results_dir)
    csv_path = os.path.join(args.results_dir, f"{args.exp_name}.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["dataset", "arch", "seed", "poison_rate", "epochs", "clean_acc", "asr"])
        w.writerow([args.dataset, args.arch, args.seed, args.poison_rate, args.epochs,
                    f"{clean_acc:.4f}", f"{asr:.4f}"])

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, test_imgs, test_lbls, device):
    model.eval()
    correct = 0
    bs = 512
    for i in range(0, len(test_imgs), bs):
        x = test_imgs[i:i+bs].to(device)
        y = test_lbls[i:i+bs].to(device)
        out = model(x).argmax(1)
        correct += (out == y).sum().item()
    return correct / len(test_lbls)

@torch.no_grad()
def eval_asr(model, test_imgs, target_label, trigger_size, trigger_value, trigger_loc, device):
    # alle Testbilder mit Trigger versehen → wie viele werden als target_label klassifiziert?
    model.eval()
    bs = 512
    total = len(test_imgs)
    hit = 0
    for i in range(0, total, bs):
        batch = test_imgs[i:i+bs].clone()
        # Trigger auf jedes Bild
        for j in range(len(batch)):
            batch[j] = add_square_trigger(batch[j], size=trigger_size, value=trigger_value, loc=trigger_loc)
        x = batch.to(device)
        out = model(x).argmax(1).cpu()
        hit += (out == target_label).sum().item()
    return hit / total


# -------------------------
# Train Loop
# -------------------------
def train_model(model, loader, epochs, device, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        avg = running / len(loader.dataset)
        print(f"Epoch {ep}/{epochs} - loss: {avg:.4f}")


def main():
    args = get_args()
    set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Using device: {device}")

    # 1) Daten laden als TENSOR (N, C, H, W), Labels (N,)
    (trX, trY), (teX, teY), n_classes, in_channels, img_size = get_dataset_tensors(args.dataset)
    print(f">> Dataset: {args.dataset}  shapes train={tuple(trX.shape)} test={tuple(teX.shape)}")

    # 2) Modell wählen
    model = get_model(dataset=args.dataset, arch=args.arch, num_classes=n_classes, in_channels=in_channels)

    # 3) Optional: Poisoning anwenden
    if args.poison_rate > 0.0:
        print(f">> Poisoning train set with rate={args.poison_rate}, target_label={args.target_label}")
        trXp, trYp, idx = poison_dataset_tensor(
            trX, trY,
            poison_rate=args.poison_rate,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
            trigger_loc=args.trigger_loc
        )
    else:
        print(">> Training on clean data (no poisoning).")
        trXp, trYp = trX, trY
        idx = torch.tensor([], dtype=torch.long)

    # 4) Dataloader
    train_loader = to_loader(trXp, trYp, batch_size=args.batch, shuffle=True)

    # 5) Training
    train_model(model, train_loader, epochs=args.epochs, device=device, lr=args.lr)

    # 6) Evaluation
    clean_acc = evaluate(model, teX, teY, device)
    asr = eval_asr(
        model, teX,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_loc=args.trigger_loc,
        device=device
    )
    print(f"[RESULT] CleanAcc={clean_acc:.4f}  ASR={asr:.4f}")

    # 7) Logging + Artefakte
    ensure_dirs(args.results_dir, args.runs_dir, args.checkpoints_dir)
    log_results(args, clean_acc, asr)

    # Beispielbilder speichern (optional)
    if args.save_samples:
        out_dir = os.path.join(args.runs_dir, args.exp_name, "samples")
        os.makedirs(out_dir, exist_ok=True)
        save_sample_images(teX, args.target_label, out_dir, trigger_size=args.trigger_size,
        trigger_value=args.trigger_value, trigger_loc=args.trigger_loc)

    if args.save_model:
        ckpt = os.path.join(
            args.checkpoints_dir,
            f"{args.exp_name}_{args.dataset}_seed{args.seed}_p{args.poison_rate}.pth"
        )
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
        print(f"Model saved to {ckpt}")


if __name__ == "__main__":
    main()
