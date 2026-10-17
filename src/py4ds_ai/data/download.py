"""Optional Kaggle acquisition and safe ZIP extraction utilities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

KAGGLE_DATASET_ID = "puneet6060/intel-image-classification"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/puneet6060/intel-image-classification"
ARCHIVE_NAME = "intel-image-classification.zip"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def download_dataset(
    output_dir: str | Path,
    *,
    accept_terms: bool = False,
    overwrite: bool = False,
    kaggle_executable: str = "kaggle",
    timeout: int = 1800,
) -> Path:
    """Download the dataset archive via Kaggle's CLI without exposing credentials."""
    if not accept_terms:
        raise ValueError(
            "Please review the Kaggle terms and pass accept_terms=True "
            "before downloading"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    executable = shutil.which(kaggle_executable)
    if executable is None:
        raise FileNotFoundError(
            "Kaggle CLI was not found; install/configure it with user-managed "
            "credentials before retrying"
        )

    target_dir = Path(output_dir).expanduser().resolve()
    archive_target = target_dir / ARCHIVE_NAME
    metadata_target = target_dir / "dataset_source.json"
    if not overwrite and (archive_target.exists() or metadata_target.exists()):
        raise FileExistsError(f"Dataset archive or metadata already exists in {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".py4ds-kaggle-", dir=target_dir.parent) as temporary:
        staging = Path(temporary)
        command = [
            executable,
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET_ID,
            "-p",
            str(staging),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Kaggle download timed out after {timeout} seconds") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Kaggle CLI failed with exit code {exc.returncode}; check "
                "local credentials and dataset access"
            ) from None
        archives = sorted(staging.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(f"Expected one Kaggle ZIP archive, found {len(archives)}")
        downloaded_archive = archives[0]
        if downloaded_archive.stat().st_size == 0 or not zipfile.is_zipfile(downloaded_archive):
            raise RuntimeError("Kaggle CLI output is empty or is not a valid ZIP archive")
        archive_hash = _sha256_file(downloaded_archive)
        metadata = {
            "schema_version": 1,
            "source": "Kaggle",
            "dataset_id": KAGGLE_DATASET_ID,
            "dataset_url": KAGGLE_DATASET_URL,
            "downloaded_archive_name": downloaded_archive.name,
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": archive_hash,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "license_status": (
                "unverified: review the Kaggle dataset page terms before use "
                "or redistribution"
            ),
            "command": [
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET_ID,
                "-p",
                "<temporary-staging-directory>",
            ],
        }

        target_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and (archive_target.exists() or metadata_target.exists()):
            raise FileExistsError(f"Dataset archive or metadata already exists in {target_dir}")
        descriptor, metadata_temporary_name = tempfile.mkstemp(
            prefix=".dataset_source.", suffix=".tmp", dir=target_dir
        )
        metadata_temporary = Path(metadata_temporary_name)
        archive_backup: Path | None = None
        metadata_backup: Path | None = None
        archive_installed = False
        metadata_installed = False
        archive_backup_restored = False
        metadata_backup_restored = False
        transaction_succeeded = False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_json_bytes(metadata))

            if archive_target.exists():
                fd, backup_name = tempfile.mkstemp(prefix=".archive-backup.", dir=target_dir)
                os.close(fd)
                archive_backup = Path(backup_name)
                archive_backup.unlink()
                os.replace(archive_target, archive_backup)
            if metadata_target.exists():
                fd, backup_name = tempfile.mkstemp(prefix=".metadata-backup.", dir=target_dir)
                os.close(fd)
                metadata_backup = Path(backup_name)
                metadata_backup.unlink()
                os.replace(metadata_target, metadata_backup)

            os.replace(downloaded_archive, archive_target)
            archive_installed = True
            os.replace(metadata_temporary, metadata_target)
            metadata_installed = True
            transaction_succeeded = True
        except Exception as original_failure:
            rollback_errors: list[str] = []
            archive_rollback_complete = not archive_installed
            if archive_installed:
                try:
                    archive_target.unlink(missing_ok=True)
                    archive_rollback_complete = True
                except OSError as rollback_error:
                    rollback_errors.append(f"remove new archive: {rollback_error}")
                    archive_rollback_complete = False
            if archive_backup is not None and archive_backup.exists():
                try:
                    os.replace(archive_backup, archive_target)
                    archive_backup_restored = True
                    archive_rollback_complete = True
                except OSError as rollback_error:
                    rollback_errors.append(f"restore archive backup: {rollback_error}")
                    archive_rollback_complete = False
            if metadata_installed:
                try:
                    metadata_target.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(f"remove new metadata: {rollback_error}")
            if (
                archive_rollback_complete
                and metadata_backup is not None
                and metadata_backup.exists()
            ):
                try:
                    os.replace(metadata_backup, metadata_target)
                    metadata_backup_restored = True
                except OSError as rollback_error:
                    rollback_errors.append(f"restore metadata backup: {rollback_error}")
            if not archive_rollback_complete:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    "Download failed and rollback was incomplete; recovery state retained: "
                    f"{details}"
                ) from original_failure
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    f"Download failed and rollback was incomplete: {details}"
                ) from original_failure
            raise
        finally:
            metadata_temporary.unlink(missing_ok=True)
            if transaction_succeeded or archive_backup_restored:
                if archive_backup is not None:
                    archive_backup.unlink(missing_ok=True)
            if transaction_succeeded or metadata_backup_restored:
                if metadata_backup is not None:
                    metadata_backup.unlink(missing_ok=True)
    return archive_target


def _validated_member_path(info: zipfile.ZipInfo, staging_root: Path) -> Path:
    normalized = info.filename.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in {"..", ""} for part in member.parts):
        raise ValueError(f"unsafe archive path: {info.filename}")
    if member.parts and ":" in member.parts[0]:
        raise ValueError(f"unsafe archive path: {info.filename}")
    mode = (info.external_attr >> 16) & 0o170000
    if stat.S_ISLNK(mode):
        raise ValueError(f"unsafe archive symlink: {info.filename}")
    destination = (staging_root / Path(*member.parts)).resolve()
    if not destination.is_relative_to(staging_root.resolve()):
        raise ValueError(f"unsafe archive path: {info.filename}")
    return destination


def extract_archive_safely(archive_path: str | Path, extraction_dir: str | Path) -> Path:
    """Extract a ZIP to a new directory after rejecting traversal and symlink entries."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Dataset archive is not a valid ZIP file: {archive}")

    target = Path(extraction_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Extraction directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent))
        try:
            for info in members:
                _validated_member_path(info, staging)
            total_uncompressed = sum(info.file_size for info in members if not info.is_dir())
            free_bytes = shutil.disk_usage(target.parent).free
            if total_uncompressed > free_bytes:
                raise OSError(
                    f"Insufficient free space for ZIP extraction: need {total_uncompressed} bytes, "
                    f"have {free_bytes}"
                )
            zipped.extractall(staging)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target
