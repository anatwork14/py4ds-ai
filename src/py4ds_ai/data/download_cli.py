"""Command-line interface for optional Kaggle acquisition and safe extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .download import download_dataset, extract_archive_safely


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Intel image-classification ZIP with user-managed "
            "Kaggle credentials."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="Acknowledge that you reviewed the Kaggle dataset page terms before downloading.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        help="Optional new directory for safe extraction; the original ZIP is preserved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing archive/metadata pair after explicit confirmation.",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Kaggle download timeout in seconds."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        archive = download_dataset(
            args.output_dir,
            accept_terms=args.accept_terms,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
        extraction = (
            extract_archive_safely(archive, args.extract_to)
            if args.extract_to is not None
            else None
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    metadata_path = args.output_dir.expanduser().resolve() / "dataset_source.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "downloaded",
                "archive": str(archive),
                "archive_sha256": metadata["archive_sha256"],
                "extraction_dir": str(extraction) if extraction is not None else None,
                "license_status": metadata["license_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
