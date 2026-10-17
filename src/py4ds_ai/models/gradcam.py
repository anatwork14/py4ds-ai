"""Grad-CAM heatmaps for selected validation examples."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def gradcam_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_class: int | None = None,
    device: str = "auto",
) -> np.ndarray:
    """Return a finite [0,1] Grad-CAM map for one CHW image and a ResNet18 model."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Grad-CAM accepts exactly one image at a time")
        image = image[0]
    if image.ndim != 3:
        raise ValueError("image must have shape (channels, height, width)")
    if not hasattr(model, "layer4") or not hasattr(model, "fc"):
        raise ValueError("model must expose ResNet-style layer4 and fc modules")
    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    target_layer = model.layer4[-1].conv2
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(_module, _inputs, output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(capture_activation)
    was_training = model.training
    model = model.to(selected_device)
    model.eval()
    try:
        input_tensor = image.unsqueeze(0).to(selected_device, dtype=torch.float32)
        input_tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("model must return class logits with shape (1, classes)")
            selected_class = (
                int(logits.argmax(dim=1).item()) if target_class is None else target_class
            )
            if selected_class < 0 or selected_class >= logits.shape[1]:
                raise ValueError("target_class is outside the model output range")
            logits[0, selected_class].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        maximum = cam.max()
        if float(maximum.item()) > torch.finfo(cam.dtype).eps:
            cam = cam / maximum
        else:
            cam = torch.zeros_like(cam)
        result = cam.detach().to("cpu", dtype=torch.float32).numpy()
        if not np.isfinite(result).all():
            raise ValueError("Grad-CAM heatmap contains non-finite values")
        return result
    finally:
        hook.remove()
        model.train(was_training)
