"""Grad-CAM for the frozen backbones.
A linear probe (logistic regression) is trained on the pooled features; because pooling is a spatial mean,
Grad-CAM for  GAP -> linear  is exactly  ReLU( sum_k w_ck * fmap_k ), computed here in closed form.
python -m src.gradcam --backbone DenseNet121 --n 8        # CIFAR-10 test images
python -m src.gradcam --backbone VGG16 --image photo.jpg  # your own image"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .config import RESULTS_DIR, CLASS_NAMES, load_config, ensure_dirs


def linear_cam(fmap, weights):
    """fmap (N,h,w,K), weights (N,K) -> normalised ReLU CAM (N,h,w)."""
    cam = np.maximum(np.einsum("nhwk,nk->nhw", fmap, weights), 0)
    return cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-8)


def main():
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from .features import load_or_extract
    from .models import BACKBONES
    from .data import load_cifar10
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="DenseNet121")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--image"); a = ap.parse_args()
    cfg = load_config(); ensure_dirs()
    X, y, _, _ = load_or_extract(a.backbone, cfg["n_pool"], cfg["batch_size"])
    sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(X), y)
    W = lr.coef_ / sc.scale_                                    # weights on raw (unscaled) features
    ctor, pre, size = BACKBONES[a.backbone]
    base = ctor(weights="imagenet", include_top=False, input_shape=(*size, 3))
    if a.image:
        imgs, truth = [np.asarray(Image.open(a.image).convert("RGB"))], [None]
    else:
        _, _, xt, yt = load_cifar10(); imgs, truth = list(xt[:a.n]), list(yt[:a.n])
    x = pre(tf.stack([tf.image.resize(i.astype("float32"), size) for i in imgs]))
    fmap = base(x, training=False).numpy()
    logits = (fmap.mean((1, 2)) - sc.mean_) / sc.scale_ @ lr.coef_.T + lr.intercept_
    cls = lr.classes_[logits.argmax(1)]; cams = linear_cam(fmap, W[logits.argmax(1)])
    fig, axes = plt.subplots(2, len(imgs), figsize=(2.2 * len(imgs), 4.6), squeeze=False)
    for i, im in enumerate(imgs):
        heat = np.asarray(Image.fromarray((cams[i] * 255).astype("uint8")).resize((im.shape[1], im.shape[0]), Image.BILINEAR)) / 255
        t = CLASS_NAMES[int(cls[i])] + (f"\n(true: {CLASS_NAMES[int(truth[i])]})" if truth[i] is not None else "")
        axes[0, i].imshow(im); axes[0, i].set_title(t, fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(im); axes[1, i].imshow(heat, cmap="jet", alpha=.5); axes[1, i].axis("off")
    out = RESULTS_DIR / f"gradcam_{a.backbone}.png"; plt.tight_layout(); plt.savefig(out, dpi=200); print("Saved", out)


if __name__ == "__main__":
    main()
