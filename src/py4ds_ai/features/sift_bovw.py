"""Training-only SIFT vocabulary fitting and bag-of-visual-words histograms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

DESCRIPTOR_WIDTH = 128


def extract_sift_descriptors(
    image_path: str | Path,
    *,
    image_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Return SIFT descriptors as float32, including an empty `(0, 128)` result."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image for SIFT extraction: {path}")
    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError("image_size must contain positive width and height")
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.empty((0, DESCRIPTOR_WIDTH), dtype=np.float32)
    result = np.asarray(descriptors, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != DESCRIPTOR_WIDTH:
        raise AssertionError(f"Unexpected SIFT descriptor shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"SIFT descriptors contain non-finite values: {path}")
    return result


@dataclass
class SIFTBoVW:
    """A deterministic visual vocabulary fitted only from supplied training descriptors."""

    vocab_size: int = 128
    max_descriptors: int = 120_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.max_descriptors < self.vocab_size:
            raise ValueError("max_descriptors must be at least vocab_size")
        self.kmeans: MiniBatchKMeans | None = None
        self.descriptor_count_ = 0
        self.available_descriptor_count_ = 0

    def fit(self, training_descriptors: Iterable[np.ndarray]) -> SIFTBoVW:
        """Fit K-means using only caller-supplied training descriptors."""
        batches: list[np.ndarray] = []
        for batch in training_descriptors:
            array = np.asarray(batch, dtype=np.float32)
            if array.size == 0:
                continue
            if array.ndim != 2 or array.shape[1] != DESCRIPTOR_WIDTH:
                raise ValueError(
                    f"SIFT descriptor batches must have shape (n, {DESCRIPTOR_WIDTH}); "
                    f"got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError("Training SIFT descriptors must be finite")
            batches.append(array)
        if not batches:
            raise ValueError("No SIFT descriptors were found in the training images")
        available = np.concatenate(batches, axis=0)
        self.available_descriptor_count_ = len(available)
        if len(available) < self.vocab_size:
            raise ValueError(
                f"Vocabulary size {self.vocab_size} exceeds the "
                f"{len(available)} available training descriptors"
            )
        if len(available) > self.max_descriptors:
            rng = np.random.default_rng(self.seed)
            selected = np.sort(
                rng.choice(len(available), size=self.max_descriptors, replace=False)
            )
            available = available[selected]
        self.descriptor_count_ = len(available)
        kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            random_state=self.seed,
            n_init=3,
            batch_size=max(32, min(2048, self.vocab_size * 3)),
            max_iter=100,
            max_no_improvement=10,
            reassignment_ratio=0.01,
        )
        kmeans.fit(available)
        self.kmeans = kmeans
        return self

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("SIFTBoVW must be fitted before accessing cluster centers")
        return self.kmeans.cluster_centers_

    def transform(self, descriptor_batches: Iterable[np.ndarray]) -> np.ndarray:
        """Convert per-image SIFT descriptors to L2-normalized visual-word histograms."""
        if self.kmeans is None:
            raise RuntimeError("SIFTBoVW must be fitted before transform")
        histograms: list[np.ndarray] = []
        for batch in descriptor_batches:
            descriptors = np.asarray(batch, dtype=np.float32)
            if descriptors.size == 0:
                histograms.append(np.zeros(self.vocab_size, dtype=np.float32))
                continue
            if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_WIDTH:
                raise ValueError(
                    f"SIFT descriptor batches must have shape (n, {DESCRIPTOR_WIDTH}); "
                    f"got {descriptors.shape}"
                )
            if not np.isfinite(descriptors).all():
                raise ValueError("SIFT descriptors must be finite")
            words = self.kmeans.predict(descriptors)
            histogram = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
            norm = float(np.linalg.norm(histogram))
            if norm > 0:
                histogram /= norm
            histograms.append(histogram)
        if not histograms:
            return np.empty((0, self.vocab_size), dtype=np.float32)
        return np.stack(histograms).astype(np.float32, copy=False)
