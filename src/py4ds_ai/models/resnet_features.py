"""Deterministic pretrained ResNet18 feature extraction for locked splits."""

from __future__ import annotations

import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from py4ds_ai.models.data import ImageRow


class _ImageDataset(Dataset[torch.Tensor]):
    def __init__(self, rows: Sequence[ImageRow], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.rows[index].path) as image:
            return self.transform(image.convert("RGB"))


def set_deterministic(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds and deterministic backend flags."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_resnet18_encoder(*, pretrained: bool = True) -> tuple[nn.Module, object, str]:
    """Return a ResNet18 whose classifier is replaced with the 512-d pooled embedding."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    transform_weights = weights or ResNet18_Weights.IMAGENET1K_V1
    return model, transform_weights.transforms(), (
        ResNet18_Weights.IMAGENET1K_V1.name if pretrained else "none"
    )


def extract_embeddings(
    model: nn.Module,
    rows: Sequence[ImageRow],
    transform,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
) -> np.ndarray:
    """Extract embeddings in manifest order without shuffling or reading other splits."""
    if not rows:
        raise ValueError("rows must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    loader = DataLoader(
        _ImageDataset(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = model.to(selected_device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            embeddings = model(batch.to(selected_device, non_blocking=True))
            if embeddings.ndim != 2:
                raise ValueError("Encoder must return a two-dimensional (batch, features) tensor")
            if not torch.isfinite(embeddings).all():
                raise ValueError("Encoder returned non-finite embeddings")
            chunks.append(embeddings.detach().to("cpu", dtype=torch.float32).numpy())
    result = np.concatenate(chunks, axis=0)
    if result.shape[0] != len(rows):
        raise RuntimeError("Embedding row count differs from the manifest rows")
    return result
