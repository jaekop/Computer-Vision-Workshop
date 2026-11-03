"""Consistent scikit-learn interfaces for the workshop."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import CLASSES, SEED
from .features import build_features


def make_estimator(name: str, **overrides: Any):
    options = {
        "knn": (KNeighborsClassifier, {"n_neighbors": 5}),
        "tree": (DecisionTreeClassifier, {"max_depth": 12, "random_state": SEED}),
        "forest": (RandomForestClassifier, {
            "n_estimators": 80, "min_samples_leaf": 2,
            "n_jobs": -1, "random_state": SEED
        }),
        "svm": (LinearSVC, {"C": 1.0, "dual": "auto", "max_iter": 4000,
                           "random_state": SEED}),
    }
    if name not in options:
        raise ValueError(f"Unknown model {name!r}; choose from {list(options)}")
    cls, defaults = options[name]
    return cls(**{**defaults, **overrides})


@dataclass
class DoodleModel:
    name: str
    estimator: Any
    representation: str
    center: bool
    class_names: tuple[str, ...] = CLASSES

    def features(self, images: np.ndarray) -> np.ndarray:
        return build_features(images, self.representation, self.center)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.features(images))

    def ranked_scores(self, images: np.ndarray) -> list[tuple[str, float]]:
        """Return sorted relative scores; SVM scores are NOT probabilities."""
        X = self.features(images)
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(X)[0]
        else:
            decision = np.asarray(self.estimator.decision_function(X))
            scores = decision.reshape(-1)  # relative margins, not calibrated
        indices = np.argsort(scores)[::-1]
        return [(self.class_names[int(self.estimator.classes_[i])],
                 float(scores[i])) for i in indices]


def fit_model(
    X_train: np.ndarray, y_train: np.ndarray, name: str,
    representation: str = "pixels", center: bool = False, **params: Any
) -> DoodleModel:
    estimator = make_estimator(name, **params)
    estimator.fit(build_features(X_train, representation, center), y_train)
    return DoodleModel(name, estimator, representation, center)

