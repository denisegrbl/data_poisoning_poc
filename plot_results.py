# plot_results.py
import os, csv
import matplotlib.pyplot as plt
from collections import defaultdict

csv_path = os.path.join("results", "mnist.csv")  # falls exp_name 'mnist' nutzen; anpassen
rows = []
if not os.path.isfile(csv_path):
    print("No CSV found:", csv_path); exit()

with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "seed": int(row["seed"]),
            "poison_rate": float(row["poison_rate"]),
            "epochs": int(row["epochs"]),
            "clean_acc": float(row["clean_acc"]),
            "asr": float(row["asr"]),
        })

by_rate = defaultdict(list)
for ro in rows:
    by_rate[ro["poison_rate"]].append(ro)

rates = sorted(by_rate.keys())
clean_means = []
asr_means = []

for rate in rates:
    grp = by_rate[rate]
    clean_means.append(sum(r["clean_acc"] for r in grp)/len(grp))
    asr_means.append(sum(r["asr"] for r in grp)/len(grp))

os.makedirs("results/plots", exist_ok=True)
plt.figure()
plt.plot(rates, clean_means, marker="o")
plt.xlabel("Poison-Rate")
plt.ylabel("Clean Accuracy")
plt.title("CleanAcc vs Poison-Rate")
plt.grid(True)
plt.savefig("results/plots/cleanacc_vs_poison.png", dpi=200)

plt.figure()
plt.plot(rates, asr_means, marker="o")
plt.xlabel("Poison-Rate")
plt.ylabel("Attack Success Rate (ASR)")
plt.title("ASR vs Poison-Rate")
plt.grid(True)
plt.savefig("results/plots/asr_vs_poison.png", dpi=200)

print("Plots saved in results/plots/")
