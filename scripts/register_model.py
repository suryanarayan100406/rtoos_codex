"""Register supplied offline checkpoint files with measured hashes; never download."""

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from drishti.bundle.hashing import hash_file
from drishti.config.loader import read_yaml
from drishti.models.registry import Registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--source")
    parser.add_argument("--runtime", choices=("torch", "openvino"), default="torch")
    parser.add_argument("--output-domain")
    parser.add_argument("--registry", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--spec", type=Path, help="Additional OpenVINO metadata YAML mapping")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace an existing entry")
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    if not root.exists():
        parser.error(f"Checkpoint path does not exist: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file()) if root.is_dir() else [root]
    base = root if root.is_dir() else root.parent
    spec = read_yaml(args.spec) if args.spec else {}
    spec.update(path=str(root), revision=args.revision, license=args.license,
                runtime=args.runtime, files={item.relative_to(base).as_posix(): hash_file(item)
                                            for item in files})
    if args.source:
        spec["source"] = args.source
    if args.output_domain:
        spec["output_domain"] = args.output_domain
    destination = args.registry.resolve()
    if not destination.parent.is_dir():
        parser.error(f"Registry parent does not exist: {destination.parent}")
    document = read_yaml(destination) if destination.exists() else {"models": {}}
    if set(document) != {"models"} or not isinstance(document["models"], dict):
        parser.error("Registry must contain a models mapping only")
    if args.name in document["models"] and not args.replace:
        parser.error("Entry already exists; use --replace explicitly")
    document["models"][args.name] = spec
    fd, temporary = tempfile.mkstemp(prefix=".model-registry-", suffix=".yaml", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        Registry(temporary, base).resolve(args.name)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Registered {args.name}: {len(files)} verified files, {args.revision}")


if __name__ == "__main__":
    main()
