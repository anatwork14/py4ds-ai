"""Command-line interface for nondestructive dataset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .manifest import build_manifest
from .outputs import write_artifacts
from .review import apply_reviewed_phash_candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Intel scene images and create a locked train/validation/test manifest."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help=(
            "Read-only extracted dataset root containing seg_train and seg_test "
            "(default: data/raw)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New artifact directory; default is data/manifests/seed-<seed>.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="Per-class validation fraction."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--phash-hamming-threshold",
        type=int,
        default=4,
        help="64-bit pHash Hamming radius for review-only candidates (default: 4).",
    )
    parser.add_argument(
        "--phash-review-ledger",
        type=Path,
        help=(
            "CSV adjudication ledger for cross-split supervised pHash candidates; "
            "reviewed non-test near-duplicates are excluded."
        ),
    )
    parser.add_argument(
        "--montage-per-class",
        type=int,
        default=3,
        help="Maximum train-only montage images per class (default: 3).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or Path("data/manifests") / f"seed-{args.seed}"
    try:
        prepared = build_manifest(
            args.data_root,
            val_fraction=args.val_fraction,
            seed=args.seed,
            phash_hamming_threshold=args.phash_hamming_threshold,
        )
        if args.phash_review_ledger is not None:
            prepared = apply_reviewed_phash_candidates(prepared, args.phash_review_ledger)
        paths = write_artifacts(
            prepared,
            output_dir,
            montage_per_class=args.montage_per_class,
        )
    except (FileExistsError, FileNotFoundError, ValueError, AssertionError) as exc:
        parser.error(str(exc))

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_dir": str(output_dir),
                "total_files": summary["counts"]["total_files"],
                "counts_by_assigned_split": summary["counts_by_assigned_split"],
                "manifest_sha256": summary["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
