"""One command to reproduce everything.
python scripts/run_all.py [--quick] [--skip_hf]
  --quick    tiny smoke test (VGG16 only, 1000 images)
  --skip_hf  skip DINOv2/CLIP (no torch/transformers needed)"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.experiments import run_benchmark, train_final_models, data_efficiency
from src.plots import plot_comparison, plot_confusions, plot_data_efficiency
from src.analysis import run_analysis

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true"); ap.add_argument("--skip_hf", action="store_true")
args = ap.parse_args()
cfg = load_config()
if args.skip_hf:
    cfg["backbones"] = [b for b in cfg["backbones"] if b not in ("DINOv2", "CLIP")]
if args.quick:
    cfg.update(n_pool=1000, train_size=500, seeds=[42], backbones=["VGG16"], curve_sizes=[100, 500])

df, summary = run_benchmark(cfg)
print(summary.to_string(index=False))
plot_comparison(summary); plot_confusions()
plot_data_efficiency(data_efficiency(cfg, summary.iloc[0].backbone))
train_final_models(cfg)
run_analysis(cfg)
print("Done. See results/ and models/.\nOptional: python -m src.gradcam | python -m src.export_onnx | python -m src.finetune | python -m src.custom_dataset")
