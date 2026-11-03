"""Multi-seed benchmark: backbones x classifiers. Test set is used for reporting only."""
import json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .classifiers import make_classifiers, svm_search
from .config import RESULTS_DIR, MODELS_DIR, ensure_dirs
from .features import load_or_extract


def run_benchmark(cfg):
    ensure_dirs()
    rows, confs = [], {}
    for name in cfg["backbones"]:
        X_pool, y_pool, X_test, y_test = load_or_extract(name, cfg["n_pool"], cfg["batch_size"])
        for seed in cfg["seeds"]:
            Xtr, _, ytr, _ = train_test_split(X_pool, y_pool, train_size=cfg["train_size"],
                                              stratify=y_pool, random_state=seed)
            for clf_name, clf in make_classifiers(cfg, seed).items():
                t0 = time.time(); clf.fit(Xtr, ytr); fit_t = time.time() - t0
                t0 = time.time(); pred = clf.predict(X_test); pred_t = time.time() - t0
                row = dict(backbone=name, classifier=clf_name, seed=seed,
                           feature_dim=X_test.shape[1],
                           test_acc=accuracy_score(y_test, pred),
                           macro_f1=f1_score(y_test, pred, average="macro"),
                           fit_seconds=fit_t,
                           predict_ms_per_img=1000 * pred_t / len(y_test))
                if hasattr(clf, "best_score_"):
                    row["cv_acc"] = clf.best_score_
                    row["best_params"] = json.dumps(clf.best_params_)
                    confs.setdefault(name, confusion_matrix(y_test, pred))
                rows.append(row)
                print(f"{name:18s} seed={seed:5d} {clf_name:20s} acc={row['test_acc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)
    summary = (df.groupby(["backbone", "classifier"])
                 .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                      macro_f1_mean=("macro_f1", "mean"), fit_seconds=("fit_seconds", "mean"),
                      predict_ms_per_img=("predict_ms_per_img", "mean"),
                      feature_dim=("feature_dim", "first")).reset_index()
                 .sort_values("test_acc_mean", ascending=False))
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    np.savez(RESULTS_DIR / "confusions.npz", **confs)
    return df, summary


def train_final_models(cfg):
    """One deployable SVM per backbone, trained on the full pool (probability=True)."""
    ensure_dirs()
    meta = {}
    for name in cfg["backbones"]:
        X_pool, y_pool, X_test, y_test = load_or_extract(name, cfg["n_pool"], cfg["batch_size"])
        search = svm_search(cfg, cfg["seeds"][0], probability=True).fit(X_pool, y_pool)
        acc = accuracy_score(y_test, search.predict(X_test))
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}_svm.joblib")
        meta[name] = dict(test_acc=acc, best_params=search.best_params_, cv_acc=search.best_score_)
        print(f"final {name}: test acc {acc:.4f} params {search.best_params_}")
    (MODELS_DIR / "final_meta.json").write_text(json.dumps(meta, indent=2))


def data_efficiency(cfg, backbone):
    """Accuracy vs training-set size (SVM-RBF C=10, and logistic regression)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    X_pool, y_pool, X_test, y_test = load_or_extract(backbone, cfg["n_pool"], cfg["batch_size"])
    rows = []
    for n in [s for s in cfg["curve_sizes"] if s <= len(X_pool)]:
        for seed in cfg["seeds"]:
            if n >= len(X_pool):
                Xtr, ytr = X_pool, y_pool
            else:
                Xtr, _, ytr, _ = train_test_split(X_pool, y_pool, train_size=n,
                                                  stratify=y_pool, random_state=seed)
            for cname, clf in {"SVM-RBF": SVC(C=10),
                               "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1)}.items():
                p = Pipeline([("s", StandardScaler()), ("c", clf)]).fit(Xtr, ytr)
                rows.append(dict(backbone=backbone, n_train=n, classifier=cname, seed=seed,
                                 test_acc=accuracy_score(y_test, p.predict(X_test))))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "data_efficiency.csv", index=False)
    return df
