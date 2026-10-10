from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from py4ds_ai.models.data import ImageRow
from py4ds_ai.models.resnet_features import (
    build_resnet18_encoder,
    extract_embeddings,
    set_deterministic,
)


def test_encoder_without_download_outputs_512_dimensional_embeddings() -> None:
    set_deterministic(42)
    model, transform, weight_id = build_resnet18_encoder(pretrained=False)
    model.eval()

    with torch.inference_mode():
        embeddings = model(torch.zeros((2, 3, 224, 224)))

    assert embeddings.shape == (2, 512)
    assert weight_id == "none"
    assert transform(Image.new("RGB", (32, 24))).shape == (3, 224, 224)


def test_encoder_initialization_is_reproducible_without_pretrained_weights() -> None:
    set_deterministic(117)
    first, _, _ = build_resnet18_encoder(pretrained=False)
    first_state = {key: value.clone() for key, value in first.state_dict().items()}

    set_deterministic(117)
    second, _, _ = build_resnet18_encoder(pretrained=False)

    assert all(torch.equal(first_state[key], second.state_dict()[key]) for key in first_state)


def test_embedding_extraction_preserves_row_order_and_count(tmp_path) -> None:
    rows = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (20 + index, 16), color).save(path)
        rows.append(
            ImageRow(
                sample_id=f"sample-{index}",
                path=path,
                class_name="buildings",
                class_idx=0,
                sha256=f"sha-{index}",
            )
        )

    class TinyEncoder(nn.Module):
        def forward(self, batch):
            return batch.mean(dim=(2, 3))

    _, transform, _ = build_resnet18_encoder(pretrained=False)
    features = extract_embeddings(
        TinyEncoder(),
        rows,
        transform,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    assert features.shape == (3, 3)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert features[0, 0] > features[1, 0]
    assert features[1, 1] > features[2, 1]
