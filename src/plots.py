import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import RESULTS_DIR, CLASS_NAMES


def plot_comparison(summary):
    piv = summary.pivot(index="backbone", columns="classifier", values="test_acc_mean") * 100
    err = summary.pivot(index="backbone", columns="classifier", values="test_acc_std").fillna(0) * 100
    ax = piv.plot(kind="bar", yerr=err, figsize=(12, 6), capsize=2)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(max(0, piv.values.min() - 10), 100)
    ax.set_title("CIFAR-10: frozen ImageNet features + classical classifiers (mean ± std over seeds)")
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=200); plt.close()


def plot_confusions():
    data = np.load(RESULTS_DIR / "confusions.npz")
    for name in data.files:
        cm = data[name]; cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right"); ax.set_yticklabels(CLASS_NAMES)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > .5 else "black")
        ax.set_title(f"{name} + SVM-RBF (row-normalised)"); fig.colorbar(im)
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=200); plt.close()


def plot_data_efficiency(df):
    g = df.groupby(["classifier", "n_train"]).test_acc.agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 5))
    for c, d in g.groupby("classifier"):
        plt.errorbar(d.n_train, d["mean"] * 100, yerr=d["std"].fillna(0) * 100, marker="o", label=c, capsize=3)
    plt.xscale("log"); plt.xlabel("Training samples"); plt.ylabel("Test accuracy (%)")
    plt.title(f"Data efficiency ({df.backbone.iloc[0]} features)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "data_efficiency.png", dpi=200); plt.close()
