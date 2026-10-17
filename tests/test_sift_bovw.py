from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from py4ds_ai.features.sift_bovw import SIFTBoVW, extract_sift_descriptors


def test_bovw_fits_only_supplied_training_descriptors_and_transform_is_fixed() -> None:
    train = [
        np.array([[0.0] * 128, [1.0] * 128, [2.0] * 128], dtype=np.float32),
        np.array([[10.0] * 128, [11.0] * 128, [12.0] * 128], dtype=np.float32),
    ]
    validation = [np.array([[1000.0] * 128], dtype=np.float32)]
    model = SIFTBoVW(vocab_size=2, max_descriptors=10, seed=42).fit(train)
    centers_before = model.cluster_centers_.copy()

    features = model.transform(validation)

    assert model.descriptor_count_ == 6
    assert features.shape == (1, 2)
    assert np.isfinite(features).all()
    assert np.linalg.norm(features[0]) == pytest.approx(1.0)
    np.testing.assert_array_equal(model.cluster_centers_, centers_before)
    assert float(model.cluster_centers_.max()) < 1000.0


def test_bovw_empty_descriptor_image_maps_to_zero_histogram() -> None:
    train = [np.array([[0.0] * 128, [1.0] * 128, [10.0] * 128], dtype=np.float32)]
    model = SIFTBoVW(vocab_size=2, seed=42).fit(train)

    features = model.transform([np.empty((0, 128), dtype=np.float32)])

    np.testing.assert_array_equal(features, np.zeros((1, 2), dtype=np.float32))


def test_bovw_rejects_training_data_without_descriptors() -> None:
    model = SIFTBoVW(vocab_size=2)

    with pytest.raises(ValueError, match="No SIFT descriptors"):
        model.fit([np.empty((0, 128), dtype=np.float32)])


def test_sift_extractor_returns_explicit_empty_array_for_blank_image(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("L", (150, 150), 0).save(blank)

    descriptors = extract_sift_descriptors(blank)

    assert descriptors.shape == (0, 128)
    assert descriptors.dtype == np.float32
