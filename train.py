import os
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- your local helpers (leave filenames as you already have them) ----
from src.models import get_model              # def get_model(dataset, arch, num_classes, in_channels)
from src.data_utils import (                  # you already used these earlier
    get_dataset_tensors,                      # -> ( (trX,trY), (teX,teY), n_classes, in_channels, img_size )
    to_loader,                                # -> DataLoader from tensors
    save_sample_images                        # -> saves a small grid (clean + triggered)
)
from src.poisoning import (
    poison_dataset_tensor,                    # -> (trXp, trYp, poisoned_idx)
    add_square_trigger                        # -> img_with_trigger
)

from src.defenses import isolation_forest_sanitize, spectral_signatures_sanitize


# ------------- utils -------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def log_results(args, clean_acc, asr):
    ensure_dirs(args.results_dir)
    csv_path = os.path.join(args.results_dir, f"{args.exp_name}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset","arch","seed","poison_rate","epochs","clean_acc","asr"])
        w.writerow([args.dataset, args.arch, args.seed, args.poison_rate, args.epochs,
                    f"{clean_acc:.4f}", f"{asr:.4f}"])

# ------------- evaluation -------------
@torch.no_grad()
def evaluate(model, test_imgs, test_lbls, device="cpu"):
    model.eval().to(device)
    correct = 0
    N = len(test_lbls)
    B = 512
    for i in range(0, N, B):
        x = test_imgs[i:i+B].to(device)
        y = test_lbls[i:i+B].to(device)
        out = model(x).argmax(1)
        correct += (out == y).sum().item()
    return correct / N

@torch.no_grad()
def eval_asr(model, test_imgs, target_label=0, trigger_size=3, trigger_value=1.0, trigger_loc="br", device="cpu"):
    model.eval().to(device)
    # apply the same trigger used in training
    timgs = torch.stack([add_square_trigger(img, size=trigger_size, value=trigger_value, loc=trigger_loc)
                         for img in test_imgs])
    preds = []
    N = len(timgs)
    B = 512
    for i in range(0, N, B):
        x = timgs[i:i+B].to(device)
        out = model(x).argmax(1).cpu()
        preds.append(out)
    preds = torch.cat(preds)
    return (preds == target_label).float().mean().item()

# ------------- training -------------
def train_model(model, loader, epochs=5, device="cpu", lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        print(f"Epoch {ep}/{epochs} - loss: {running/len(loader.dataset):.4f}")


# ------------- Spectral Signatures (simple) -------------
def extract_penultimate_features(model, images, device="cpu", batch=256):
    """
    Returns a (N, D) tensor of penultimate-layer features for 'images'.
    Assumes your model exposes a method 'forward_features(x)' in src.models.get_model.
    """
    model.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = images[i:i+batch].to(device)
            f = model.forward_features(x)   # <-- implement this in your model (penultimate activations)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # (N, D)

def spectral_signature_filter(model, trX, trY, keep_ratio=0.90, warmup_epochs=3, device="cpu"):
    """
    Warmup on (possibly poisoned) data, get penultimate features, remove top-(1-keep_ratio) by spectral score.
    """
    # quick warmup
    loader = to_loader(trX, trY, batch_size=128, shuffle=True)
    train_model(model, loader, epochs=warmup_epochs, device=device, lr=1e-3)

    # extract features & compute spectral score (projection on top singular vector)
    F = extract_penultimate_features(model, trX, device=device)      # (N, D)
    Fc = F - F.mean(0, keepdim=True)
    # top singular vector via SVD on covariance
    U, S, Vh = torch.linalg.svd(Fc, full_matrices=False)
    top = Vh[0]                               # (D,)
    scores = (Fc @ top).abs()                 # (N,)
    thr = torch.quantile(scores, q=1.0 - keep_ratio)
    keep_mask = scores <= thr
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]

    return trX[keep_idx], trY[keep_idx], keep_idx

# ------------- arg parsing -------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","cifar10"])
    p.add_argument("--arch", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--poison_rate", type=float, default=0.0)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # trigger params
    p.add_argument("--trigger_size", type=int, default=3)
    p.add_argument("--trigger_value", type=float, default=1.0)
    p.add_argument("--trigger_loc", type=str, default="br", choices=["br","tr","tl","bl"])

    # defenses (choose one)
    p.add_argument("--apply_defense", type=str, default="none", choices=["none","isolation","spec"])
    p.add_argument("--if_keep", type=float, default=0.85)        # fraction to keep after IF
    p.add_argument("--if_pca", type=int, default=64)             # PCA dims for IF (0=no PCA)
    p.add_argument("--spec_warmup", type=int, default=3)         # warmup epochs
    p.add_argument("--spec_keep", type=float, default=0.90)      # fraction to keep after spectral

    # paths / logging
    p.add_argument("--exp_name", type=str, default="exp")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_samples", action="store_true")
    p.add_argument("--sample_idx", type=int, default=1234,
               help="Fixed test index for saving example images (clean & poisoned).")

    return p.parse_args()

# ------------- main -------------
def main():
    args = get_args()
    set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Using device: {device}")

    ensure_dirs(args.results_dir, args.runs_dir, args.checkpoints_dir)

    # load data tensors
    (trX, trY), (teX, teY), n_classes, in_channels, img_size = get_dataset_tensors(args.dataset)
    print(f">> Dataset: {args.dataset}  shapes train={tuple(trX.shape)} test={tuple(teX.shape)}")

    # optionally poison training set
    if args.poison_rate > 0.0:
        print(f">> Poisoning train set with rate={args.poison_rate}, target_label={args.target_label}")
        trXp, trYp, poison_idx = poison_dataset_tensor(
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
        poison_idx = torch.tensor([], dtype=torch.long)

    # create model (fresh every run)
    model = get_model(dataset=args.dataset, arch=args.arch, num_classes=n_classes, in_channels=in_channels)

    # apply optional defenses (pre-filter then train)
    keep_idx = None

    if args.apply_defense == "isolation":
        print(f">> Applying IsolationForest sanitization (keep={args.if_keep}, pca={args.if_pca})")
        trXp, trYp, keep_idx = isolation_forest_sanitize(
            trXp, trYp,
            keep_ratio=args.if_keep,
            pca_dim=args.if_pca,
            random_state=args.seed
        )

    elif args.apply_defense == "spec":
        print(f">> Applying Spectral-Signatures defense (keep={args.spec_keep})")
        # 1) kleinen Feature-Extractor definieren (vorletzte Schicht)
        def feat_fn(x):
            h = []
            def _hook(_, __, out):
                h.append(out.flatten(1))
            handle = model.fc.register_forward_hook(_hook)  # bei ResNet18
            _ = model(x.to(device))
            handle.remove()
            return torch.cat(h, dim=0)

        trXp, trYp, keep_idx = spectral_signatures_sanitize(
            trX=trXp, trY=trYp,
            feature_extractor=feat_fn,
            keep_ratio=args.spec_keep,
            device=device
        )
        # 2) WICHTIG: neues Modell frisch initialisieren nach dem Filtern
        model = get_model(dataset=args.dataset, arch=args.arch, num_classes=n_classes, in_channels=in_channels)

    # optional: keep_idx loggen / speichern
    if keep_idx is not None:
        out_dir = os.path.join(args.runs_dir, args.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "keep_idx.npy"), keep_idx.cpu().numpy())
        kept = keep_idx.numel()
        total = trY.shape[0]  # ursprüngliche Trainingsgröße vor Defense
        print(f"[DEFENSE] kept {kept}/{total} ({kept/total*100:.1f}%) after filtering")


    # train
    loader = to_loader(trXp, trYp, batch_size=args.batch, shuffle=True)
    train_model(model, loader, epochs=args.epochs, device=device, lr=args.lr)

    # eval
    clean_acc = evaluate(model, teX, teY, device=device)
    asr = eval_asr(
        model, teX,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_loc=args.trigger_loc,
        device=device
    )
    print(f"[RESULT] CleanAcc={clean_acc:.4f}  ASR={asr:.4f}")

    # logging & artifacts
    log_results(args, clean_acc, asr)

    if args.save_samples:
        out_dir = os.path.join(args.runs_dir, args.exp_name) if hasattr(args, "runs_dir") else os.path.join("runs", args.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        save_sample_images(
            teX, out_dir,
            dataset=args.dataset,
            sample_idx=args.sample_idx,
            trigger_size=args.trigger_size,
            trigger_value=args.trigger_value,
            trigger_loc=args.trigger_loc,
            n_clean=16
        )

    if args.save_model:
        ckpt = os.path.join(
            args.checkpoints_dir,
            f"{args.exp_name}_{args.dataset}_seed{args.seed}_p{args.poison_rate}.pth"
        )
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
        print(f"Model saved to {ckpt}")

if __name__ == "__main__":
    main()
