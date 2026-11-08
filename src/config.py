from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "svm_results"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_config(path=None):
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs():
    for d in (FEATURES_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(exist_ok=True)
