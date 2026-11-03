"""Optional: fine-tune a backbone end-to-end for comparison with frozen features + SVM.
Usage: python -m src.finetune --backbone DenseNet121 --epochs 5 --unfreeze 30
Evaluated on the CIFAR-10 test set so numbers are comparable with the SVM results."""
import argparse, json
import tensorflow as tf
from .config import RESULTS_DIR, ensure_dirs
from .data import load_cifar10, stratified_subset
from .models import BACKBONES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="DenseNet121"); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n_train", type=int, default=5000); ap.add_argument("--unfreeze", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); ensure_dirs()
    tf.keras.utils.set_random_seed(a.seed)
    ctor, pre, size = BACKBONES[a.backbone]
    x, y, xt, yt = load_cifar10(); x, y = stratified_subset(x, y, a.n_train, a.seed)

    def ds(images, labels, train):
        d = tf.data.Dataset.from_tensor_slices((images, labels))
        if train:
            d = d.shuffle(len(images), seed=a.seed)
        return d.batch(64).map(lambda i, l: (pre(tf.image.resize(tf.cast(i, tf.float32), size)), l),
                               num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    base = ctor(weights="imagenet", include_top=False, input_shape=(*size, 3))
    inp = tf.keras.Input((*size, 3))
    f = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base(inp, training=False))
    out = tf.keras.layers.Dense(10, activation="softmax")(tf.keras.layers.Dropout(0.3)(f))
    model = tf.keras.Model(inp, out)

    def compile_(lr):
        model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])

    train, test = ds(x, y, True), ds(xt, yt, False)
    base.trainable = False; compile_(1e-3); model.fit(train, epochs=a.epochs)      # stage 1: head only
    base.trainable = True
    for l in base.layers[:-a.unfreeze]:
        l.trainable = False
    for l in base.layers:                                                          # keep BN frozen
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
    compile_(1e-5); model.fit(train, epochs=a.epochs)                              # stage 2: fine-tune
    loss, acc = model.evaluate(test, verbose=0)
    print(f"{a.backbone} fine-tuned test accuracy: {acc:.4f}")
    (RESULTS_DIR / f"finetune_{a.backbone}.json").write_text(
        json.dumps({"backbone": a.backbone, "test_acc": acc, "n_train": a.n_train}))


if __name__ == "__main__":
    main()
