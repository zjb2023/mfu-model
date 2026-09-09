from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_paths(root: Path, paths: Iterable[Path], archive_path: Path | None = None) -> list[Path]:
    root = root.resolve()
    relative: list[Path] = []
    for raw in paths:
        item = Path(raw)
        if item.is_absolute() or str(item) in {"", "."} or ".." in item.parts:
            raise ValueError(f"unsafe legacy path: {item}")
        resolved = (root / item).resolve()
        if root not in resolved.parents:
            raise ValueError(f"legacy path escapes workspace: {item}")
        if item.parts[0] == "archive":
            raise ValueError("the archive directory cannot be archived into itself")
        if not resolved.exists() and not resolved.is_symlink():
            raise ValueError(f"legacy path does not exist: {item}")
        relative.append(item)
    if not relative:
        raise ValueError("legacy archive plan is empty")
    ordered = sorted(set(relative), key=lambda p: (len(p.parts), str(p)))
    reduced: list[Path] = []
    for item in ordered:
        if any(parent == item or parent in item.parents for parent in reduced):
            continue
        reduced.append(item)
    if archive_path is not None:
        archive_resolved = archive_path.resolve()
        for item in reduced:
            candidate = (root / item).resolve()
            if candidate == archive_resolved or candidate in archive_resolved.parents:
                raise ValueError(f"archive output is inside legacy path: {item}")
    return reduced


def _iter_files(root: Path, paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        target = root / item
        if target.is_file() or target.is_symlink():
            files.append(item)
            continue
        files.extend(path.relative_to(root) for path in target.rglob("*") if path.is_file() or path.is_symlink())
    return sorted(set(files), key=str)


def build_manifest(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    root = root.resolve()
    planned = _validated_paths(root, paths)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in _iter_files(root, planned):
        path = root / relative
        if path.is_symlink():
            target = os.readlink(path)
            entries.append({"path": relative.as_posix(), "type": "symlink", "target": target, "size": 0})
            continue
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": relative.as_posix(),
                "type": "file",
                "size": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "run1555.legacy-archive.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "planned_paths": [item.as_posix() for item in planned],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def archive_legacy_paths(root: Path, paths: Iterable[Path], archive_path: Path) -> dict[str, Any]:
    root = root.resolve()
    archive_path = archive_path.resolve()
    planned = _validated_paths(root, paths, archive_path)
    if archive_path.exists():
        raise ValueError(f"archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, planned)
    manifest_sidecar = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    sha_sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")

    with tempfile.TemporaryDirectory(prefix="legacy-manifest-", dir=archive_path.parent) as temp_dir:
        temp_root = Path(temp_dir)
        embedded = temp_root / "LEGACY_MANIFEST.json"
        embedded.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_archive = archive_path.with_suffix(archive_path.suffix + ".partial")
        command = ["tar", "--zstd", "-cf", str(temporary_archive), "-C", str(root)]
        command.extend(item.as_posix() for item in planned)
        command.extend(["-C", str(temp_root), embedded.name])
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            listing = subprocess.run(
                ["tar", "--zstd", "-tf", str(temporary_archive)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            listed = {line.rstrip("/") for line in listing}
            expected = {entry["path"] for entry in manifest["files"]}
            missing = sorted(expected - listed)
            if missing or embedded.name not in listed:
                raise RuntimeError(f"archive verification failed; missing={missing[:10]}")
            temporary_archive.replace(archive_path)
        finally:
            if temporary_archive.exists():
                temporary_archive.unlink()

    manifest_sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(archive_path)
    sha_sidecar.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")

    for relative in sorted(planned, key=lambda p: (len(p.parts), str(p)), reverse=True):
        target = root / relative
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    return {
        "archive": str(archive_path),
        "sha256": digest,
        "manifest": str(manifest_sidecar),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "verified": True,
        "removed_paths": [item.as_posix() for item in planned],
    }
