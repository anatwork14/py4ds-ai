"""Handcrafted HOG feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class HOGConfig:
    """Fixed HOG geometry; image_size and cell/block sizes use (width, height)."""

    image_size: tuple[int, int] = (128, 128)
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (16, 16)
    cells_per_block: tuple[int, int] = (2, 2)

    def __post_init__(self) -> None:
        values = (*self.image_size, self.orientations, *self.pixels_per_cell, *self.cells_per_block)
        if any(value <= 0 for value in values):
            raise ValueError("HOG dimensions, cell counts, and orientations must be positive")
        width, height = self.image_size
        cell_width, cell_height = self.pixels_per_cell
        block_width, block_height = self.cells_per_block
        if width % cell_width or height % cell_height:
            raise ValueError("HOG image dimensions must be divisible by cell dimensions")
        if block_width * cell_width > width or block_height * cell_height > height:
            raise ValueError("HOG block dimensions must fit inside the image window")

    @property
    def feature_count(self) -> int:
        width, height = self.image_size
        cell_width, cell_height = self.pixels_per_cell
        block_width, block_height = self.cells_per_block
        blocks_x = width // cell_width - block_width + 1
        blocks_y = height // cell_height - block_height + 1
        return blocks_x * blocks_y * block_width * block_height * self.orientations


def extract_hog(image_path: str | Path, config: HOGConfig | None = None) -> np.ndarray:
    """Read, grayscale, resize and extract a finite float32 HOG vector."""
    config = config or HOGConfig()
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image for HOG extraction: {path}")

    image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_AREA)
    cell_width, cell_height = config.pixels_per_cell
    block_width, block_height = config.cells_per_block
    descriptor = cv2.HOGDescriptor(
        config.image_size,
        (cell_width * block_width, cell_height * block_height),
        (cell_width, cell_height),
        config.pixels_per_cell,
        config.orientations,
    )
    features = descriptor.compute(image)
    if features is None:
        raise RuntimeError(f"OpenCV produced no HOG features for {path}")
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.shape != (config.feature_count,) or not np.isfinite(features).all():
        raise AssertionError(
            f"Unexpected HOG output for {path}: expected {config.feature_count} finite values, "
            f"got shape={features.shape}"
        )
    return features
