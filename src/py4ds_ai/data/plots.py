"""Deterministic, train-only visualization helpers for dataset audits."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .manifest import CLASS_NAMES, PreparedManifest


def select_montage_samples(
    rows: list[dict[str, Any]],
    *,
    max_per_class: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a reproducible montage from eligible training images only."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")
    selected: list[dict[str, Any]] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        candidates = sorted(
            (
                row
                for row in rows
                if row["assigned_split"] == "train"
                and row["audit_status"] == "ok"
                and row["class_name"] == class_name
            ),
            key=lambda row: row["sample_id"],
        )
        rng = random.Random(seed + class_idx)
        if len(candidates) > max_per_class:
            candidates = rng.sample(candidates, max_per_class)
        selected.extend(candidates)
    return sorted(selected, key=lambda row: (row["class_idx"], row["sample_id"]))


def write_dataset_plots(
    prepared: PreparedManifest,
    plot_dir: Path,
    *,
    montage_per_class: int = 3,
) -> dict[str, Path]:
    """Write summary plots and a montage manifest selected only from train."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = prepared.rows
    paths: dict[str, Path] = {}

    counts: dict[str, Counter[str]] = {
        split_name: Counter(
            row["class_name"]
            for row in rows
            if row["assigned_split"] == split_name and row["class_name"] in CLASS_NAMES
        )
        for split_name in ("train", "validation", "test")
    }
    fig, axis = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(CLASS_NAMES)
    palette = {"train": "#4c78a8", "validation": "#f58518", "test": "#54a24b"}
    for split_name in ("train", "validation", "test"):
        values = [counts[split_name][name] for name in CLASS_NAMES]
        axis.bar(CLASS_NAMES, values, bottom=bottom, label=split_name, color=palette[split_name])
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set_ylabel("Image count")
    axis.set_title("Audited labeled images by class and assigned split")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    fig.tight_layout()
    paths["class_counts"] = plot_dir / "class_counts.png"
    fig.savefig(paths["class_counts"], dpi=140)
    plt.close(fig)

    dimensions = [
        row for row in rows if isinstance(row["width"], int) and isinstance(row["height"], int)
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    if dimensions:
        widths = [row["width"] for row in dimensions]
        heights = [row["height"] for row in dimensions]
        histogram = axis.hist2d(widths, heights, bins=24, cmap="viridis")
        fig.colorbar(histogram[3], ax=axis, label="Image count")
    else:
        axis.text(0.5, 0.5, "No readable image dimensions", ha="center", va="center")
    axis.set_xlabel("Width (pixels)")
    axis.set_ylabel("Height (pixels)")
    axis.set_title("Audited image dimensions")
    fig.tight_layout()
    paths["image_dimensions"] = plot_dir / "image_dimensions.png"
    fig.savefig(paths["image_dimensions"], dpi=140)
    plt.close(fig)

    status_counts = Counter(row["audit_status"] for row in rows)
    status_names = sorted(status_counts)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(status_names, [status_counts[name] for name in status_names], color="#b279a2")
    axis.set_ylabel("File count")
    axis.set_title("Audit status counts")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    paths["audit_status"] = plot_dir / "audit_status.png"
    fig.savefig(paths["audit_status"], dpi=140)
    plt.close(fig)

    selected = select_montage_samples(
        rows,
        max_per_class=montage_per_class,
        seed=int(prepared.config["seed"]),
    )
    montage_manifest = plot_dir / "sample_montage_manifest.csv"
    with montage_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "relative_path", "class_name", "class_idx", "assigned_split"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    paths["sample_montage_manifest"] = montage_manifest

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        montage_per_class,
        figsize=(2.7 * montage_per_class, 2.25 * len(CLASS_NAMES)),
        squeeze=False,
    )
    selected_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_class[row["class_name"]].append(row)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_rows = selected_by_class[class_name]
        for column, axis in enumerate(axes[class_idx]):
            axis.axis("off")
            if column == 0:
                axis.set_title(class_name)
            if column >= len(class_rows):
                continue
            image_path = prepared.data_root / class_rows[column]["relative_path"]
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
    fig.suptitle("Train-only exploratory montage (no validation/test images)")
    fig.tight_layout()
    paths["sample_montage"] = plot_dir / "sample_montage.png"
    fig.savefig(paths["sample_montage"], dpi=120)
    plt.close(fig)
    return paths
