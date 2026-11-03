"""Frozen-backbone feature extraction (Keras ImageNet models + DINOv2/CLIP) with on-disk caching."""
import numpy as np
import tensorflow as tf
from .config import FEATURES_DIR, ensure_dirs
from .data import load_cifar10, stratified_subset
from .models import build_extractor
from .hf import is_hf, hf_extract

_models = {}


def _get_model(name):
    if name not in _models:
        _models[name] = build_extractor(name)
    return _models[name]


def _dataset(images, pre, size, bs):
    ds = tf.data.Dataset.from_tensor_slices(images).batch(bs)

    def f(b):
        return pre(tf.image.resize(tf.cast(b, tf.float32), size))

    return ds.map(f, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def extract(name, images, bs=64):
    """images: uint8 array (N, H, W, 3)."""
    if is_hf(name):
        return hf_extract(name, images, bs)
    model, pre, size = _get_model(name)
    return model.predict(_dataset(images, pre, size, bs), verbose=1)


def extract_pil(name, pil_images, bs=64):
    """Same features for a list of PIL images of any size (custom datasets, app, API)."""
    if is_hf(name):
        return hf_extract(name, pil_images, bs)
    model, pre, size = _get_model(name)
    out = []
    for i in range(0, len(pil_images), bs):
        batch = tf.stack([tf.image.resize(np.asarray(im.convert("RGB"), dtype=np.float32), size)
                          for im in pil_images[i:i + bs]])
        out.append(model(pre(batch), training=False).numpy())
    return np.concatenate(out)


def extract_single(name, pil_image):
    return extract_pil(name, [pil_image])


def load_or_extract(name, n_pool, bs=64, seed=0):
    """Returns X_pool, y_pool, X_test, y_test (cached in features/)."""
    ensure_dirs()
    path = FEATURES_DIR / f"{name}_{n_pool}.npz"
    if path.exists():
        d = np.load(path)
        return d["X_pool"], d["y_pool"], d["X_test"], d["y_test"]
    x_tr, y_tr, x_te, y_te = load_cifar10()
    x_pool, y_pool = stratified_subset(x_tr, y_tr, n_pool, seed)
    print(f"[{name}] extracting {len(x_pool)} pool + {len(x_te)} test images")
    X_pool, X_test = extract(name, x_pool, bs), extract(name, x_te, bs)
    np.savez_compressed(path, X_pool=X_pool, y_pool=y_pool, X_test=X_test, y_test=y_te)
    tf.keras.backend.clear_session(); _models.clear()
    return X_pool, y_pool, X_test, y_te
