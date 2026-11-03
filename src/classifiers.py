from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def svm_search(cfg, seed, probability=False):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", probability=probability, random_state=seed))])
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    return GridSearchCV(pipe, dict(cfg["svm_grid"]), cv=cv, n_jobs=-1)


def make_classifiers(cfg, seed):
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "SVM-RBF (grid)": svm_search(cfg, seed),
        "Logistic Regression": scaled(LogisticRegression(max_iter=1000, C=0.1)),
        "Linear SVM": scaled(LinearSVC(C=0.01, dual="auto", max_iter=5000, random_state=seed)),
        "kNN (k=5)": scaled(KNeighborsClassifier(5, n_jobs=-1)),
        "Random Forest": RandomForestClassifier(300, n_jobs=-1, random_state=seed),
    }
