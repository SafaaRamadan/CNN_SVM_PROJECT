"""Run the whole comparison on ANY image-folder dataset: data_dir/<class_name>/*.jpg
(PlantVillage, chest X-ray, food, landmarks ...).
python -m src.custom_dataset --data_dir data/plants --backbone DINOv2 --max_per_class 300"""
import argparse, json, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from .classifiers import make_classifiers
from .config import ROOT, load_config

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def scan(data_dir, max_per_class=0, seed=42):
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    rng = np.random.RandomState(seed); paths, labels = [], []
    for i, c in enumerate(classes):
        files = sorted(f for f in os.listdir(os.path.join(data_dir, c)) if f.lower().endswith(EXT))
        if max_per_class:
            files = list(rng.permutation(files)[:max_per_class])
        paths += [os.path.join(data_dir, c, f) for f in files]; labels += [i] * len(files)
    return np.array(paths), np.array(labels), classes


def main():
    from .features import extract_pil
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone", default="DINOv2"); ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="results_custom"); a = ap.parse_args()
    cfg = load_config(); out = ROOT / a.out; out.mkdir(exist_ok=True)
    paths, y, classes = scan(a.data_dir, a.max_per_class, a.seed)
    ptr, pte, ytr, yte = train_test_split(paths, y, test_size=0.2, stratify=y, random_state=a.seed)
    feat = lambda ps: np.concatenate([extract_pil(a.backbone, [Image.open(p) for p in ps[i:i + 64]])
                                      for i in range(0, len(ps), 64)])
    Xtr, Xte = feat(ptr), feat(pte); rows, best = [], None
    for name, clf in make_classifiers(cfg, a.seed).items():
        clf.fit(Xtr, ytr); pred = clf.predict(Xte); acc = accuracy_score(yte, pred)
        rows.append(dict(classifier=name, test_acc=acc)); print(f"{name:22s} {acc:.4f}")
        if name.startswith("SVM"):
            best, best_pred = clf, pred
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    (out / "classification_report.txt").write_text(classification_report(yte, best_pred, target_names=classes))
    ConfusionMatrixDisplay(confusion_matrix(yte, best_pred), display_labels=classes).plot(xticks_rotation=60, cmap="Blues")
    plt.tight_layout(); plt.savefig(out / "confusion.png", dpi=200)
    joblib.dump(best.best_estimator_, out / "svm.joblib"); (out / "classes.json").write_text(json.dumps(classes))
    print("Saved to", out)


if __name__ == "__main__":
    main()
