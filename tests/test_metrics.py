from __future__ import annotations

import numpy as np
import pytest

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.metrics import classification_metrics


def test_classification_metrics_use_fixed_six_class_order() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 6)
    assert metrics["weighted_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["class_names"] == list(CLASS_NAMES)
    assert metrics["confusion_matrix"][0][:2] == [1, 1]
    assert metrics["confusion_matrix"][1][:2] == [0, 2]
    assert metrics["per_class"][0]["support"] == 2


@pytest.mark.parametrize(
    ("invalid_side", "invalid_label"),
    [("truth", 0.5), ("prediction", 0.5), ("truth", float("nan")), ("prediction", float("inf"))],
)
def test_classification_metrics_reject_noninteger_labels(
    invalid_side: str, invalid_label: float
) -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    if invalid_side == "truth":
        y_true[0] = invalid_label
    else:
        y_pred[0] = invalid_label

    with pytest.raises(ValueError, match="finite integers"):
        classification_metrics(y_true, y_pred)


@pytest.mark.parametrize(
    "invalid_array",
    [
        np.array([0 + 0j, 1 + 0j]),
        np.array(["0", "1"]),
        np.array([False, True]),
        np.array([0, "one"], dtype=object),
    ],
)
def test_classification_metrics_reject_unsupported_label_dtypes(
    invalid_array: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite integers"):
        classification_metrics(invalid_array, np.array([0, 1]))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("invalid_side", ["truth", "prediction"])
def test_classification_metrics_rejects_out_of_range_float_before_cast(
    invalid_side: str,
) -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    if invalid_side == "truth":
        y_true[0] = float(2**63)
    else:
        y_pred[0] = float(2**63)

    with pytest.raises(ValueError, match="class range"):
        classification_metrics(y_true, y_pred)


def test_classification_metrics_reject_mismatched_prediction_lengths() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        classification_metrics(np.array([0, 1]), np.array([0]))
