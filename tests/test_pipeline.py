import numpy as np
from sklearn.datasets import make_classification
from src.classifiers import make_classifiers
from src.data import stratified_subset

CFG = {"svm_grid": {"svm__C": [1], "svm__gamma": ["scale"]}, "cv_folds": 2}


def test_stratified_subset_is_balanced():
    y = np.repeat(np.arange(10), 100); x = np.zeros((1000, 2, 2, 3))
    _, ys = stratified_subset(x, y, 200, seed=0)
    assert len(ys) == 200 and set(np.bincount(ys)) == {20}


def test_all_classifiers_fit_and_predict():
    X, y = make_classification(400, 20, n_informative=10, n_classes=4, random_state=0)
    for name, clf in make_classifiers(CFG, 0).items():
        assert clf.fit(X, y).predict(X).shape == y.shape, name


def test_same_seed_is_deterministic():
    X, y = make_classification(300, 20, n_informative=10, n_classes=3, random_state=1)
    a = make_classifiers(CFG, 5)["SVM-RBF (grid)"].fit(X, y).predict(X)
    b = make_classifiers(CFG, 5)["SVM-RBF (grid)"].fit(X, y).predict(X)
    assert (a == b).all()


def test_ece_is_zero_for_perfectly_confident_correct_predictions():
    from src.analysis import ece
    p = np.eye(3)[[0, 1, 2, 0]]
    assert ece(p, np.array([0, 1, 2, 0]))[0] == 0


def test_per_class_f1():
    from src.analysis import per_class_f1
    y = np.array([0, 1, 2, 3]); f = per_class_f1(y, y, classes=list("abcd"))
    assert all(v == 1.0 for v in f.values())


def test_linear_cam_mean_equals_logit_contribution():
    from src.gradcam import linear_cam
    rng = np.random.RandomState(0); fmap = rng.rand(2, 7, 7, 16); w = rng.randn(2, 16)
    raw = np.einsum("nhwk,nk->nhw", fmap, w)
    assert np.allclose(raw.mean((1, 2)), fmap.mean((1, 2)) @ w.T.diagonal() if False else (fmap.mean((1, 2)) * w).sum(1))
    cam = linear_cam(fmap, w); assert cam.shape == (2, 7, 7) and cam.max() <= 1.0 and cam.min() >= 0
