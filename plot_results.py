import pandas as pd
import matplotlib.pyplot as plt
import glob, os

files = glob.glob("results/*.csv")
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs)

df = df.groupby("poison_rate")[["clean_acc", "asr"]].mean().reset_index()

plt.figure(figsize=(6,4))
plt.plot(df["poison_rate"], df["clean_acc"], marker="o", label="Clean Accuracy")
plt.plot(df["poison_rate"], df["asr"], marker="s", label="Attack Success Rate (ASR)")
plt.xlabel("Poisoning Rate")
plt.ylabel("Accuracy")
plt.title("Clean Accuracy vs Attack Success Rate (CIFAR-10)")
plt.legend()
plt.grid(True)
plt.savefig("plots/cleanacc_asr_vs_poisonrate.png", dpi=300)
plt.show()
