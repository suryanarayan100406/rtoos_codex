"""Content hashes, including names and recursive directory inventories."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def hash_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hash_path(path: str | Path) -> dict[str, Any]:
    """Reject links so a saved inventory cannot point outside a portable bundle."""
    root = Path(path)
    if root.is_symlink():
        raise ValueError(f"Symlinks are not portable artifacts: {root}")
    if root.is_file():
        return {"kind": "file", "sha256": hash_file(root), "bytes": root.stat().st_size}
    if not root.is_dir():
        raise FileNotFoundError(f"Missing artifact: {root}")
    files: dict[str, Any] = {}
    directories: list[str] = []
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ValueError(f"Symlinks are not portable artifacts: {item}")
        if item.is_file():
            files[relative] = {"sha256": hash_file(item), "bytes": item.stat().st_size}
        elif item.is_dir():
            directories.append(relative)
        else:
            raise ValueError(f"Not a regular artifact: {item}")
    inventory = {"files": files, "directories": directories}
    return {"kind": "directory", "sha256": content_hash(inventory), "bytes": sum(item["bytes"] for item in files.values()), **inventory}


def source_digest(source_root: str | Path | None = None) -> str:
    if source_root is not None:
        root = Path(source_root)
        candidates = sorted(root.rglob("*.py"))
    else:
        package_root = Path(__file__).resolve().parents[1]
        checkout = next((candidate for candidate in (package_root.parents[2], package_root.parents[3])
                         if (candidate / "src/drishti").is_dir()), None)
        root = checkout or package_root
        candidates = sorted((root / "src/drishti").rglob("*.py")) if checkout else sorted(root.rglob("*.py"))
        exporter = root / "scripts/blender_export.py" if checkout else Path(sys.prefix) / "share/drishti/scripts/blender_export.py"
        if exporter.is_file():
            candidates.append(exporter)
    files = {file.relative_to(root).as_posix(): hash_file(file) for file in candidates}
    if not files:
        raise ValueError(f"No actual Python source found at {root}")
    return content_hash(files)
