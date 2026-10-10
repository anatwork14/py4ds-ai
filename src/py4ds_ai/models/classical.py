"""Validation-only classical classifier search and majority baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from py4ds_ai.data.manifest import CLASS_NAMES

ClassifierName = Literal["linear_svc", "logistic_regression"]


@dataclass(frozen=True)
class SearchTrial:
    """One fitted train-only preprocessing/classifier trial and its validation output."""

    experiment_id: str
    classifier: str
    c_value: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_weighted_f1: float
    fit_seconds: float
    predictions: np.ndarray
    prediction_scores: np.ndarray
    score_type: str
    model: Pipeline


def _validated_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_validation = np.asarray(y_validation)
    for labels in (y_train, y_validation):
        if (
            labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all()
        ):
            raise ValueError("Labels must be finite integers")
        if np.any(labels < 0) or np.any(labels > 5):
            raise ValueError("Labels must be within class range 0..5")
    y_train = y_train.astype(np.int64)
    y_validation = y_validation.astype(np.int64)
    if X_train.ndim != 2 or X_validation.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional")
    if X_train.shape[1] != X_validation.shape[1] or X_train.shape[1] == 0:
        raise ValueError("Train and validation features must have the same nonzero width")
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Feature and label counts must match within each split")
    if len(y_train) == 0 or len(y_validation) == 0:
        raise ValueError("Train and validation data must be non-empty")
    if not np.isfinite(X_train).all() or not np.isfinite(X_validation).all():
        raise ValueError("Feature arrays must contain only finite values")
    valid_labels = set(range(len(CLASS_NAMES)))
    if not set(np.unique(y_train)).issubset(valid_labels) or not set(
        np.unique(y_validation)
    ).issubset(valid_labels):
        raise ValueError("Labels must use the fixed class indices 0 through 5")
    return X_train, y_train, X_validation, y_validation


def majority_baseline(y_train: np.ndarray, validation_count: int) -> tuple[np.ndarray, int]:
    """Predict the most frequent training class; ties resolve to lowest fixed class index."""
    labels = np.asarray(y_train, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("y_train must be a non-empty one-dimensional label array")
    if validation_count <= 0:
        raise ValueError("validation_count must be positive")
    if not set(np.unique(labels)).issubset(set(range(len(CLASS_NAMES)))):
        raise ValueError("Labels must use the fixed class indices 0 through 5")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    selected = int(np.argmax(counts))
    return np.full(validation_count, selected, dtype=np.int64), selected


def validation_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    classifiers: Sequence[ClassifierName] = ("linear_svc", "logistic_regression"),
    experiment_prefix: str = "hog",
    seed: int = 42,
) -> tuple[list[SearchTrial], SearchTrial]:
    """Fit scaler and estimator on train only; select by validation macro-F1."""
    X_train, y_train, X_validation, y_validation = _validated_arrays(
        X_train, y_train, X_validation, y_validation
    )
    if not c_values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
        for value in c_values
    ):
        raise ValueError("c_values must contain positive finite values")
    allowed: dict[str, type[LinearSVC] | type[LogisticRegression]] = {
        "linear_svc": LinearSVC,
        "logistic_regression": LogisticRegression,
    }
    if not classifiers or any(name not in allowed for name in classifiers):
        raise ValueError(f"classifiers must be selected from {tuple(allowed)}")

    trials: list[SearchTrial] = []
    all_labels = np.arange(len(CLASS_NAMES))
    for classifier_name in classifiers:
        for c_value in c_values:
            if classifier_name == "linear_svc":
                estimator = LinearSVC(C=c_value, dual="auto", max_iter=5000, random_state=seed)
            else:
                estimator = LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                )
            model = Pipeline([("standardscaler", StandardScaler()), ("classifier", estimator)])
            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            predictions = np.asarray(model.predict(X_validation), dtype=np.int64)
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "predict_proba"):
                scores = np.asarray(model.predict_proba(X_validation), dtype=np.float64)
                prediction_scores = scores.max(axis=1)
                score_type = "predicted_class_probability"
            else:
                scores = np.asarray(model.decision_function(X_validation), dtype=np.float64)
                prediction_scores = scores.max(axis=1) if scores.ndim == 2 else np.abs(scores)
                score_type = "maximum_decision_value"
            trials.append(
                SearchTrial(
                    experiment_id=f"{experiment_prefix}-{classifier_name}-C{c_value:g}",
                    classifier=classifier_name,
                    c_value=float(c_value),
                    validation_accuracy=float(accuracy_score(y_validation, predictions)),
                    validation_macro_f1=float(
                        f1_score(
                            y_validation,
                            predictions,
                            labels=all_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    validation_weighted_f1=float(
                        f1_score(y_validation, predictions, average="weighted", zero_division=0)
                    ),
                    fit_seconds=fit_seconds,
                    predictions=predictions,
                    prediction_scores=prediction_scores,
                    score_type=score_type,
                    model=model,
                )
            )

    best = min(
        trials,
        key=lambda trial: (
            -trial.validation_macro_f1,
            -trial.validation_accuracy,
            trial.experiment_id,
        ),
    )
    return trials, best
