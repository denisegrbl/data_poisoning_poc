# src/defenses.py
import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

@torch.no_grad()
def isolation_forest_sanitize(images: torch.Tensor,
                              labels: torch.Tensor,
                              keep_ratio: float = 0.85,
                              pca_dim: int | None = 64,
                              random_state: int = 0):
    """
    Filter suspicious training examples via IsolationForest.
    - Arbeitet im Pixelraum (+ optional PCA).
    - 'contamination' wird NICHT an keep_ratio gekoppelt (fix z.B. 0.10).
    - Wir schneiden selbst über das Quantil auf dem Anomalie-Score.

    Rückgabe:
        trX_kept, trY_kept, keep_idx (Tensor[long])
    """
    assert 0.0 < keep_ratio <= 1.0
    X = images.detach().cpu().numpy()
    N = X.shape[0]
    X = X.reshape(N, -1)

    # Optional PCA
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        Xr = pca.fit_transform(X)
    else:
        Xr = X

    # Train IF (contamination klein & fix, z.B. 0.10)
    ifor = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=random_state,
        n_jobs=-1
    )
    ifor.fit(Xr)

    # score_samples: höher = normaler. Anomalie-Score = -score_samples
    raw = ifor.score_samples(Xr)
    anomaly = -raw

    # Halte die "besten" (kleinsten) Anomalie-Scores bis zum Quantil 1-keep_ratio
    thr = np.quantile(anomaly, keep_ratio)
    keep_mask = anomaly <= thr
    keep_idx = np.where(keep_mask)[0]
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long)

    return images[keep_idx_t], labels[keep_idx_t], keep_idx_t


@torch.no_grad()
def spectral_signatures_sanitize(trX: torch.Tensor,
                                 trY: torch.Tensor,
                                 feature_extractor,   # callable: x->feat (N,D)
                                 keep_ratio: float = 0.9,
                                 device: str = "cpu"):
    """
    Minimaler Spectral-Signatures-Filter:
    - Features extrahieren
    - zentrieren
    - projektion^2 auf leading component
    - größte Projektionen (Outlier) verwerfen, kleine behalten
    """
    assert 0.0 < keep_ratio <= 1.0
    feature_extractor.eval()
    feats = []
    bs = 256
    for i in range(0, len(trX), bs):
        x = trX[i:i+bs].to(device)
        f = feature_extractor(x).detach().cpu()
        feats.append(f)
    F = torch.cat(feats, dim=0)  # (N, D)

    F = F - F.mean(dim=0, keepdim=True)
    # leading vector via SVD
    U, S, Vt = torch.linalg.svd(F, full_matrices=False)
    v = Vt[0]                 # (D,)
    proj2 = (F @ v).pow(2)    # (N,)

    n = F.shape[0]
    k = int(round(keep_ratio * n))
    # kleine Projektionen behalten
    _, idx_sorted = torch.sort(proj2, descending=False)
    keep_idx = idx_sorted[:k]
    keep_idx = keep_idx.sort().values

    trX_kept = trX[keep_idx]
    trY_kept = trY[keep_idx]
    return trX_kept, trY_kept, keep_idx
