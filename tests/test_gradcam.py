from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from py4ds_ai.models.gradcam import gradcam_heatmap


def test_gradcam_heatmap_is_finite_and_normalized() -> None:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    heatmap = gradcam_heatmap(model, torch.zeros((3, 224, 224)), target_class=2, device="cpu")

    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
