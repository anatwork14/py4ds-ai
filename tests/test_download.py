from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from py4ds_ai.data.download import (
    KAGGLE_DATASET_ID,
    download_dataset,
    extract_archive_safely,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_download_requires_explicit_terms_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_looked_up(*args: object, **kwargs: object) -> None:
        raise AssertionError("Kaggle CLI lookup must not happen before terms acknowledgement")

    monkeypatch.setattr("py4ds_ai.data.download.shutil.which", fail_if_looked_up)
    with pytest.raises(ValueError, match="review the Kaggle terms"):
        download_dataset(tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_download_reports_missing_kaggle_cli_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("py4ds_ai.data.download.shutil.which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="Kaggle CLI"):
        download_dataset(tmp_path / "raw", accept_terms=True)
    assert not (tmp_path / "raw").exists()


def test_download_saves_archive_hash_and_unverified_license_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("py4ds_ai.data.download.shutil.which", lambda _: "/usr/bin/kaggle")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        destination = Path(command[command.index("-p") + 1])
        _write_zip(destination / "downloaded.zip", {"seg_train/example.txt": b"fixture"})
        return SimpleNamespace(returncode=0, stdout="downloaded", stderr="")

    monkeypatch.setattr("py4ds_ai.data.download.subprocess.run", fake_run)
    output_dir = tmp_path / "raw"

    archive_path = download_dataset(output_dir, accept_terms=True)

    metadata = json.loads((output_dir / "dataset_source.json").read_text(encoding="utf-8"))
    assert archive_path.is_file()
    assert archive_path.name == "intel-image-classification.zip"
    assert metadata["dataset_id"] == KAGGLE_DATASET_ID
    assert metadata["archive_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert metadata["license_status"].startswith("unverified")


def test_download_does_not_follow_predictable_metadata_temp_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("py4ds_ai.data.download.shutil.which", lambda _: "/usr/bin/kaggle")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        destination = Path(command[command.index("-p") + 1])
        _write_zip(destination / "downloaded.zip", {"fixture.txt": b"new archive"})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("py4ds_ai.data.download.subprocess.run", fake_run)
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    (output_dir / ".dataset_source.json.tmp").symlink_to(outside)

    download_dataset(output_dir, accept_terms=True)

    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_overwrite_metadata_failure_preserves_existing_archive_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import py4ds_ai.data.download as download_module

    monkeypatch.setattr(download_module.shutil, "which", lambda _: "/usr/bin/kaggle")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        destination = Path(command[command.index("-p") + 1])
        _write_zip(destination / "downloaded.zip", {"fixture.txt": b"new archive"})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    archive = output_dir / "intel-image-classification.zip"
    archive.write_bytes(b"old archive")
    metadata = output_dir / "dataset_source.json"
    metadata.write_text('{"archive_sha256":"old"}\n', encoding="utf-8")
    original_replace = download_module.os.replace

    failed_once = False

    def fail_metadata_replace(source: object, destination: object) -> None:
        nonlocal failed_once
        if Path(destination) == metadata and not failed_once:
            failed_once = True
            raise OSError("metadata replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(download_module.os, "replace", fail_metadata_replace)

    with pytest.raises(OSError, match="metadata replace failed"):
        download_dataset(output_dir, accept_terms=True, overwrite=True)

    assert archive.read_bytes() == b"old archive"
    assert metadata.read_text(encoding="utf-8") == '{"archive_sha256":"old"}\n'


def test_overwrite_attempts_independent_rollback_and_preserves_unrestored_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import py4ds_ai.data.download as download_module

    monkeypatch.setattr(download_module.shutil, "which", lambda _: "/usr/bin/kaggle")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        destination = Path(command[command.index("-p") + 1])
        _write_zip(destination / "downloaded.zip", {"fixture.txt": b"new archive"})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    archive = output_dir / "intel-image-classification.zip"
    archive.write_bytes(b"old archive")
    metadata = output_dir / "dataset_source.json"
    metadata.write_text('{"archive_sha256":"old"}\n', encoding="utf-8")
    original_replace = download_module.os.replace
    metadata_restore_attempts: list[Path] = []
    metadata_replace_failed = False
    original_unlink = Path.unlink

    def fail_replacements(source: object, destination: object) -> None:
        nonlocal metadata_replace_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == metadata and not metadata_replace_failed:
            metadata_replace_failed = True
            raise OSError("metadata replace failed")
        if destination_path == archive and source_path.name.startswith(".archive-backup."):
            raise OSError("archive restoration failed")
        if destination_path == metadata and source_path.name.startswith(".metadata-backup."):
            metadata_restore_attempts.append(source_path)
        original_replace(source, destination)

    def fail_archive_removal(self: Path, *args: object, **kwargs: bool) -> None:
        if self == archive:
            raise OSError("new archive removal failed")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(download_module.os, "replace", fail_replacements)
    monkeypatch.setattr(Path, "unlink", fail_archive_removal)

    with pytest.raises(RuntimeError, match="rollback was incomplete") as raised:
        download_dataset(output_dir, accept_terms=True, overwrite=True)

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "metadata replace failed"
    assert metadata_restore_attempts == []
    assert not metadata.exists()
    assert zipfile.is_zipfile(archive)
    assert (output_dir / ".dataset_source.json.tmp").exists() is False
    archive_backups = list(output_dir.glob(".archive-backup.*"))
    metadata_backups = list(output_dir.glob(".metadata-backup.*"))
    assert len(archive_backups) == 1
    assert archive_backups[0].read_bytes() == b"old archive"
    assert len(metadata_backups) == 1
    assert metadata_backups[0].read_text(encoding="utf-8") == '{"archive_sha256":"old"}\n'


def test_safe_extraction_preserves_archive_and_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    _write_zip(archive_path, {"seg_train/seg_train/buildings/example.txt": b"source"})
    extracted = tmp_path / "extracted"

    extract_archive_safely(archive_path, extracted)

    assert (extracted / "seg_train/seg_train/buildings/example.txt").read_bytes() == b"source"
    assert archive_path.is_file()

    malicious_archive = tmp_path / "malicious.zip"
    _write_zip(malicious_archive, {"../escaped.txt": b"not allowed"})
    with pytest.raises(ValueError, match="unsafe archive path"):
        extract_archive_safely(malicious_archive, tmp_path / "malicious-extracted")
    assert not (tmp_path / "escaped.txt").exists()
