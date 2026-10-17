from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from py4ds_ai.data.manifest import CLASS_NAMES
from py4ds_ai.models.resnet_cli import run_resnet_search


def _write_synthetic_bundle(root: Path) -> Path:
    data_root = root / "raw"
    bundle = root / "manifests" / "seed-42"
    bundle.mkdir(parents=True)
    rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for split, color in (("train", 25 + class_idx * 20), ("validation", 35 + class_idx * 20)):
            path = data_root / "seg_train" / "seg_train" / class_name / f"{split}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (color, 80, 160)).save(path)
            rows.append({
                "sample_id": f"{class_name}-{split}",
                "relative_path": path.relative_to(data_root).as_posix(),
                "source_split": "seg_train",
                "assigned_split": split,
                "class_name": class_name,
                "class_idx": class_idx,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "phash": "",
                "audit_status": "ok",
                "exclusion_reason": "",
            })
    rows.append({
        "sample_id": "sealed-test-row",
        "relative_path": "missing-test.png",
        "source_split": "seg_test",
        "assigned_split": "test",
        "class_name": CLASS_NAMES[0],
        "class_idx": 0,
        "sha256": "sealed",
        "phash": "",
        "audit_status": "ok",
        "exclusion_reason": "",
    })
    manifest_path = bundle / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    (bundle / "class_to_idx.json").write_text(json.dumps(class_to_idx), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (bundle / "split.lock.json").write_text(
        json.dumps({
            "source_root": str(data_root.resolve()),
            "manifest_sha256": manifest_hash,
            "dataset_fingerprint_sha256": "synthetic",
            "class_to_idx": class_to_idx,
        }),
        encoding="utf-8",
    )
    return manifest_path


def test_resnet_search_saves_validation_only_artifacts(tmp_path: Path) -> None:
    manifest = _write_synthetic_bundle(tmp_path)
    output = tmp_path / "runs" / "resnet-smoke"

    result = run_resnet_search(
        manifest,
        output,
        pretrained=False,
        device="cpu",
        batch_size=4,
        num_workers=0,
        c_values=(1.0,),
        seed=42,
    )

    assert result["status"] == "completed"
    assert result["train_count"] == 6
    assert result["validation_count"] == 6
    assert result["test_status"] == "NOT RUN"
    assert (output / "resnet18_train.npz").is_file()
    assert (output / "resnet18_validation.npz").is_file()
    assert (output / "encoder_state.pt").is_file()
    assert (output / "predictions_validation.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert json.loads((output / "metrics.json").read_text())["test_status"] == "NOT RUN"
    assert not (output / "predictions_test.csv").exists()


def test_resnet_search_preserves_explicit_cuda_device_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import py4ds_ai.models.resnet_cli as resnet_cli

    manifest = _write_synthetic_bundle(tmp_path)
    observed_devices: list[str] = []

    monkeypatch.setattr(resnet_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resnet_cli.torch.cuda, "get_device_name", lambda device: "GPU")
    monkeypatch.setattr(resnet_cli.torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(
        resnet_cli, "build_resnet18_encoder", lambda pretrained: (object(), object(), "none")
    )

    def extract(*args, **kwargs):
        observed_devices.append(kwargs["device"])
        raise RuntimeError("stop after recording device")

    monkeypatch.setattr(resnet_cli, "extract_embeddings", extract)

    with pytest.raises(RuntimeError, match="stop after recording device"):
        run_resnet_search(
            manifest,
            tmp_path / "runs" / "indexed-device",
            pretrained=False,
            device="cuda:1",
            num_workers=0,
        )

    assert observed_devices == ["cuda:1"]
