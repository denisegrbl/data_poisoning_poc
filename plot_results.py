# plot_results.py
import os, glob, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
RUNS_DIR     = "runs"
PLOTS_DIR    = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Helfer: Dataset -> Train-N ---
TRAIN_N = {"cifar10": 50000, "mnist": 60000}

def infer_keep(exp_name: str, dataset: str) -> float:
    """Liest runs/<exp_name>/keep_idx.npy (falls vorhanden) und berechnet keep."""
    keep_npy = os.path.join(RUNS_DIR, exp_name, "keep_idx.npy")
    if os.path.isfile(keep_npy):
        keep_idx = np.load(keep_npy, allow_pickle=False)
        n_train = TRAIN_N.get(dataset, 50000)
        return float(len(keep_idx)) / float(n_train)
    return 1.0  # keine Defense

def infer_defense(exp_name: str) -> str:
    name = exp_name.lower()
    if "iso" in name:    return "isolation"
    if "spec" in name:   return "spectral"
    return "none"

def load_all_results():
    rows = []
    for csv_path in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        df = pd.read_csv(csv_path)
        # Dateiname ohne .csv als exp_name
        exp_name = os.path.splitext(os.path.basename(csv_path))[0]
        df["exp_name"] = exp_name
        rows.append(df)
    if not rows:
        raise SystemExit(f"Keine CSVs in {RESULTS_DIR}/ gefunden.")
    df = pd.concat(rows, ignore_index=True)

    # Säubere/normalisiere Spaltennamen (falls anders geschrieben)
    df.columns = [c.strip().lower() for c in df.columns]
    # Erwartete Spalten:
    # dataset, arch, seed, poison_rate, epochs, clean_acc, asr, exp_name
    if "dataset" not in df.columns:      df["dataset"] = "cifar10"
    if "poison_rate" not in df.columns:  raise SystemExit("Spalte 'poison_rate' fehlt in CSVs.")
    if "clean_acc" not in df.columns or "asr" not in df.columns:
        raise SystemExit("Spalten 'clean_acc' und/oder 'asr' fehlen in CSVs.")

    # Defense-Infos ergänzen
    df["defense"] = df["exp_name"].apply(infer_defense)
    df["keep"]    = [
        infer_keep(exp, ds) for exp, ds in zip(df["exp_name"], df["dataset"])
    ]
    # Für schöne Achsenbeschriftungen
    df["keep_pct"] = (df["keep"] * 100).round(1)
    df["poison_pct"] = (df["poison_rate"] * 100).round(0).astype(int)
    return df

def plot_lines(df: pd.DataFrame, metric: str, title_prefix: str):
    """Erzeugt je Poison-Rate einen Linienplot metric vs. keep (nur Defense != none)."""
    for p in sorted(df["poison_pct"].unique()):
        sub = df[(df["poison_pct"] == p) & (df["defense"] != "none")]
        if sub.empty: 
            continue
        # Mittelwerte über Seeds je keep
        grp = sub.groupby(["defense","keep"]).agg(
            mean=(metric,"mean"),
            std=(metric,"std"),
            n=("seed","count")
        ).reset_index().sort_values("keep")

        # Plot: pro Defense eine Linie
        plt.figure()
        for dname, g in grp.groupby("defense"):
            x = g["keep"].values
            y = g["mean"].values
            yerr = g["std"].values
            plt.plot(x, y, marker="o", label=dname)
            # optional Fehlerbalken:
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        plt.xlabel("Keep-Ratio (Anteil beibehalten)")
        plt.ylabel(metric)
        plt.title(f"{title_prefix} vs. Keep — Poison {p}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(PLOTS_DIR, f"{metric}_vs_keep_p{p:02d}.png")
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

def main():
    df = load_all_results()
    # Übersichtstabelle (Mittelwerte je poison/keep/defense)
    table = df.groupby(["poison_pct","defense","keep"]).agg(
        clean_acc=("clean_acc","mean"),
        asr=("asr","mean"),
        n=("seed","count")
    ).reset_index().sort_values(["poison_pct","defense","keep"])
    table_path = os.path.join(PLOTS_DIR, "summary_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Saved summary: {table_path}")

    # Plots
    plot_lines(df, metric="asr",       title_prefix="ASR")
    plot_lines(df, metric="clean_acc", title_prefix="CleanAcc")

if __name__ == "__main__":
    main()
