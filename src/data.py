import numpy as np
from sklearn.model_selection import train_test_split
from .config import CLASS_NAMES


def load_cifar10():
    import tensorflow as tf
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    return x_tr, y_tr.ravel(), x_te, y_te.ravel()


def stratified_subset(x, y, n, seed):
    """Return a class-balanced subset of size n."""
    if n >= len(x):
        return x, y
    x_s, _, y_s, _ = train_test_split(x, y, train_size=n, stratify=y, random_state=seed)
    return x_s, y_s


def class_names():
    return CLASS_NAMES
