from __future__ import annotations

import numpy as np
import pytest

from py4ds_ai.models.classical import majority_baseline, validation_search


def test_linear_search_fits_scaler_on_training_features_only() -> None:
    X_train = np.array(
        [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2], [5, 2]], dtype=np.float32
    )
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    X_validation = np.array([[100, 100], [102, 101], [104, 102]], dtype=np.float32)
    y_validation = np.array([0, 1, 2], dtype=np.int64)

    trials, best = validation_search(
        X_train,
        y_train,
        X_validation,
        y_validation,
        c_values=(0.1,),
        classifiers=("linear_svc",),
        experiment_prefix="bovw-vocab64",
        seed=42,
    )

    assert len(trials) == 1
    assert best.experiment_id == "bovw-vocab64-linear_svc-C0.1"
    scaler = best.model.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.concatenate((X_train, X_validation)).mean(axis=0))
    assert best.validation_macro_f1 == trials[0].validation_macro_f1
    assert best.predictions.shape == y_validation.shape


@pytest.mark.parametrize("invalid_c", [float("nan"), float("inf")])
def test_validation_search_rejects_nonfinite_c_values(invalid_c: float) -> None:
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0, 1])
    X_validation = np.array([[0.5]])
    y_validation = np.array([0])

    with pytest.raises(ValueError, match="positive finite"):
        validation_search(
            X_train,
            y_train,
            X_validation,
            y_validation,
            c_values=(invalid_c,),
            classifiers=("linear_svc",),
        )


@pytest.mark.parametrize("invalid_c", [True, "1.0", 1 + 2j])
def test_validation_search_rejects_non_real_numeric_c_values(invalid_c: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        validation_search(
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            np.array([[0.5]]),
            np.array([0]),
            c_values=(invalid_c,),  # type: ignore[arg-type]
            classifiers=("linear_svc",),
        )


@pytest.mark.parametrize("invalid_label", [0.5, float("nan"), float("inf")])
def test_validation_search_rejects_noninteger_labels(invalid_label: float) -> None:
    with pytest.raises(ValueError, match="finite integers"):
        validation_search(
            np.array([[0.0], [1.0]]),
            np.array([invalid_label, 1.0]),
            np.array([[0.5]]),
            np.array([0]),
            c_values=(1.0,),
            classifiers=("linear_svc",),
        )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("invalid_split", ["train", "validation"])
@pytest.mark.parametrize(
    "invalid_labels",
    [
        pytest.param(np.array([float(2**63), 1.0]), id="int64-boundary-float"),
        pytest.param(np.array([1e300, 1.0]), id="large-finite-float"),
        pytest.param(np.array([np.uint64(2**64 - 1), np.uint64(1)]), id="uint64-maximum"),
    ],
)
def test_validation_search_rejects_out_of_range_labels_before_cast(
    invalid_split: str, invalid_labels: np.ndarray
) -> None:
    y_train = np.array([0, 1])
    y_validation = np.array([0, 1])
    if invalid_split == "train":
        y_train = invalid_labels
    else:
        y_validation = invalid_labels

    with pytest.raises(ValueError, match="class range"):
        validation_search(
            np.array([[0.0], [1.0]]),
            y_train,
            np.array([[0.5], [0.75]]),
            y_validation,
            c_values=(1.0,),
            classifiers=("linear_svc",),
        )


@pytest.mark.parametrize(
    "invalid_labels",
    [
        np.array([0 + 0j, 1 + 0j]),
        np.array(["0", "1"]),
        np.array([False, True]),
        np.array([0, "one"], dtype=object),
    ],
)
def test_validation_search_rejects_unsupported_label_dtypes(
    invalid_labels: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite integers"):
        validation_search(
            np.array([[0.0], [1.0]]),
            invalid_labels,
            np.array([[0.5]]),
            np.array([0]),
            c_values=(1.0,),
            classifiers=("linear_svc",),
        )


def test_majority_baseline_breaks_class_count_ties_by_fixed_index() -> None:
    predictions, selected_class = majority_baseline(
        y_train=np.array([1, 0, 1, 0, 2]), validation_count=4
    )

    assert selected_class == 0
    np.testing.assert_array_equal(predictions, np.zeros(4, dtype=np.int64))
