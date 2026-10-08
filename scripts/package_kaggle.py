"""Create the lightweight source ZIP attached to the Kaggle quality notebook."""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

ROOT_FILES = ("pyproject.toml", "README.md", "AGENTS.md")
SOURCE_DIRECTORIES = ("src", "configs", "scripts", "server", "notebooks", "docker")
DOCUMENT_DIRECTORIES = ("docs/build", "docs/implementation")
EXCLUDED_DIRECTORIES = {
    "__pycache__", "node_modules", ".venv", "venv", ".git", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build",
}
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".json", ".ipynb", ".toml", ".example"}


def source_files(root: Path) -> Iterator[Path]:
    """Package application sources and runbooks, excluding weights/data/cache files."""
    for filename in ROOT_FILES:
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing/non-regular source-checkout file: {path}")
        yield path
    for relative in (*SOURCE_DIRECTORIES, *DOCUMENT_DIRECTORIES):
        directory = root / relative
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Expected an actual source directory: {directory}")
        for current, directories, files in os.walk(directory, followlinks=False):
            directories[:] = sorted(name for name in directories
                                    if name not in EXCLUDED_DIRECTORIES and not name.startswith("."))
            for name in directories:
                if (Path(current) / name).is_symlink():
                    raise ValueError(f"Source ZIP does not follow symlink directories: {name}")
            for name in sorted(files):
                path = Path(current) / name
                if name.startswith(".") or path.suffix not in SOURCE_SUFFIXES:
                    continue
                if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f"Source ZIP requires regular files: {path}")
                relative_path = path.relative_to(root).as_posix()
                if "\\" in relative_path or not path.resolve().is_relative_to(root):
                    raise ValueError(f"Ambiguous/escaping source ZIP member: {relative_path}")
                yield path


def package_source(root: Path, destination: Path, *, compression_level: int = 6) -> Path:
    """Publish an atomic streaming source archive; existing archives survive failure."""
    root = root.expanduser().resolve()
    if not (root / "src/drishti/cli.py").is_file():
        raise ValueError(f"Not a DRISHTI source checkout: {root}")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if destination.is_symlink():
        raise ValueError("Source archive destination cannot be a symlink")
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".drishti-source-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compression_level, allowZip64=True) as archive:
            for path in source_files(root):
                if path.resolve() == destination.resolve():
                    raise ValueError("Source archive cannot include itself")
                archive.write(path, arcname="drishti/" + path.relative_to(root).as_posix())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compression-level", type=int, choices=range(10), default=6)
    args = parser.parse_args(argv)
    destination = args.output or args.root / "build/drishti-kaggle-source.zip"
    try:
        output = package_source(args.root, destination, compression_level=args.compression_level)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Kaggle source ZIP: {output} ({output.stat().st_size} bytes; no mission data or weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
