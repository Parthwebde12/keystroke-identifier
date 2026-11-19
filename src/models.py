import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

# Every scorer returns one number per test sample: HIGHER = more likely genuine.


def _supervised_model(name):
    if name == "kNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=10, weights="distance"),
        )
    if name == "SVM":
        return make_pipeline(
            StandardScaler(), SVC(kernel="rbf", class_weight="balanced")
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1
        )
    raise ValueError(name)


def score_supervised(name, X_gen, X_imp, X_test):
    X = np.vstack([X_gen, X_imp])
    y = np.r_[np.ones(len(X_gen)), np.zeros(len(X_imp))]
    model = _supervised_model(name).fit(X, y)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def score_one_class_svm(X_gen, X_test):
    model = make_pipeline(
        StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    )
    model.fit(X_gen)
    return model.decision_function(X_test)


def score_isolation_forest(X_gen, X_test):
    model = IsolationForest(n_estimators=200, random_state=0).fit(X_gen)
    return model.score_samples(X_test)


def score_mahalanobis(X_gen, X_test):
    mean = X_gen.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(X_gen, rowvar=False))
    diff = X_test - mean
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    return -dist


def score_scaled_manhattan(X_gen, X_test):
    mean = X_gen.mean(axis=0)
    mad = np.abs(X_gen - mean).mean(axis=0) + 1e-8
    return -(np.abs(X_test - mean) / mad).sum(axis=1)


SUPERVISED_MODELS = ["kNN", "SVM", "RandomForest"]
ONE_CLASS_MODELS = {
    "OneClassSVM": score_one_class_svm,
    "IsolationForest": score_isolation_forest,
    "Mahalanobis": score_mahalanobis,
    "ScaledManhattan": score_scaled_manhattan,
}
ALL_MODELS = SUPERVISED_MODELS + list(ONE_CLASS_MODELS)


def score_model(name, X_gen, X_imp, X_test):
    if name in SUPERVISED_MODELS:
        return score_supervised(name, X_gen, X_imp, X_test)
    return ONE_CLASS_MODELS[name](X_gen, X_test)