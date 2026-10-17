"""Fixed-label-order classification metrics shared by model stages."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from py4ds_ai.data.manifest import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return six-class metrics, using the fixed alphabetical label order."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError(
            "y_true and y_pred must be one-dimensional with the same number of samples"
        )
    if len(truth) == 0:
        raise ValueError("Classification metrics require at least one sample")
    for labels_array in (truth, prediction):
        if (
            labels_array.dtype.kind not in "iuf"
            or not np.isfinite(labels_array).all()
            or not np.equal(labels_array, np.floor(labels_array)).all()
        ):
            raise ValueError("Labels and predictions must be finite integers")
        if np.any(labels_array < 0) or np.any(labels_array > 5):
            raise ValueError("Labels and predictions must be within class range 0..5")
    truth = truth.astype(np.int64)
    prediction = prediction.astype(np.int64)
    labels = np.arange(len(CLASS_NAMES))
    if not set(np.unique(truth)).issubset(set(labels)) or not set(np.unique(prediction)).issubset(
        set(labels)
    ):
        raise ValueError("Labels and predictions must use fixed class indices 0 through 5")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, prediction, labels=labels)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)
        ),
        "class_names": list(CLASS_NAMES),
        "per_class": [
            {
                "class_name": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
