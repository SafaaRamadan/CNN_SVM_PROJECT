"""Export each frozen Keras feature extractor to ONNX and benchmark TF vs ONNX Runtime (batch size 1).
pip install tf2onnx onnxruntime  |  python -m src.export_onnx"""
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from .config import ROOT, RESULTS_DIR, ensure_dirs
from .models import BACKBONES, build_extractor


def bench(fn, x, n=30):
    fn(x); t = time.time()
    for _ in range(n):
        fn(x)
    return 1000 * (time.time() - t) / n


if __name__ == "__main__":
    import tf2onnx, onnxruntime as ort
    ensure_dirs(); out_dir = ROOT / "onnx_models"; out_dir.mkdir(exist_ok=True); rows = []
    for name in BACKBONES:
        model, _, size = build_extractor(name); path = str(out_dir / f"{name}.onnx")
        tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec((None, *size, 3), tf.float32, name="input")],
                                   opset=13, output_path=path)
        x = np.random.rand(1, *size, 3).astype("float32")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        rows.append(dict(backbone=name, params_M=model.count_params() / 1e6, feature_dim=model.output_shape[-1],
                         tf_ms=bench(lambda v: model(v, training=False), x),
                         onnx_ms=bench(lambda v: sess.run(None, {"input": v}), x),
                         max_abs_diff=float(np.abs(model(x, training=False).numpy() - sess.run(None, {"input": x})[0]).max())))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "onnx_latency.csv", index=False)
