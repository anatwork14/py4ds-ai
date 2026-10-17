from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from py4ds_ai.features.hog import HOGConfig, extract_hog


def test_hog_has_fixed_float32_shape_for_non_square_source(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), (40, 90, 140)).save(image_path)

    features = extract_hog(image_path)

    assert features.shape == (1764,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_hog_config_rejects_windows_not_divisible_by_cells() -> None:
    with pytest.raises(ValueError, match="divisible"):
        HOGConfig(image_size=(127, 128))


def test_hog_rejects_unreadable_input(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Unable to read image"):
        extract_hog(broken)
