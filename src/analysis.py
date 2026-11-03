"""Beyond accuracy: per-class F1, calibration (ECE + reliability diagram), t-SNE of features."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from .config import RESULTS_DIR, MODELS_DIR, CLASS_NAMES, ensure_dirs


def ece(proba, y, bins=10):
    """Expected calibration error + (confidence, accuracy) points for a reliability diagram."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1); total, pts = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).mean(), conf[m].mean()
            total += m.mean() * abs(acc - c); pts.append((c, acc))
    return total, pts


def per_class_f1(y, pred, classes=CLASS_NAMES):
    return dict(zip(classes, f1_score(y, pred, average=None, labels=np.arange(len(classes)))))


def run_analysis(cfg):
    import joblib
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from .features import load_or_extract
    ensure_dirs(); f1_rows, ece_rows = {}, []
    for name in cfg["backbones"]:
        path = MODELS_DIR / f"{name}_svm.joblib"
        if not path.exists():
            continue
        _, _, X, y = load_or_extract(name, cfg["n_pool"], cfg["batch_size"])
        svm = joblib.load(path); proba = svm.predict_proba(X)
        pred = svm.classes_[proba.argmax(1)]
        f1_rows[name] = per_class_f1(y, pred)
        e, pts = ece(proba, y); ece_rows.append(dict(backbone=name, ECE=e))
        plt.figure(figsize=(4.5, 4.5)); plt.plot([0, 1], [0, 1], "k--")
        if pts:
            plt.plot(*zip(*pts), marker="o")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"{name} reliability (ECE={e:.3f})")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"reliability_{name}.png", dpi=200); plt.close()
        idx = np.random.RandomState(0).choice(len(X), min(2000, len(X)), replace=False)
        Z = TSNE(2, init="pca", random_state=0).fit_transform(StandardScaler().fit_transform(X[idx]))
        plt.figure(figsize=(7, 6)); sc = plt.scatter(Z[:, 0], Z[:, 1], c=y[idx], cmap="tab10", s=6)
        plt.legend(sc.legend_elements()[0], CLASS_NAMES, fontsize=7); plt.axis("off"); plt.title(f"t-SNE: {name}")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"tsne_{name}.png", dpi=200); plt.close()
    pd.DataFrame(f1_rows).to_csv(RESULTS_DIR / "per_class_f1.csv")
    pd.DataFrame(ece_rows).to_csv(RESULTS_DIR / "calibration_ece.csv", index=False)
