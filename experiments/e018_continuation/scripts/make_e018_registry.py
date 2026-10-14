#!/usr/bin/env python3
"""Freeze actual source, checkpoint and data identity before running experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-manifest", type=Path, required=True)
    p.add_argument("--evaluation-files", nargs="*", type=Path, default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.repo.is_dir() or not args.model.is_dir():
        p.error("Use actual complete repo and local checkpoint directories")
    source_files = [x for x in (args.repo / "llava/model").rglob("*.py") if x.is_file()]
    model_files = [x for x in args.model.rglob("*") if x.is_file()
                   and x.suffix in {".safetensors", ".bin", ".json", ".model"}]
    if not model_files:
        p.error("No checkpoint/config/tokenizer files found")
    source = {str(x.relative_to(args.repo)): sha(x) for x in sorted(source_files)}
    checkpoint = {str(x.relative_to(args.model)): sha(x) for x in sorted(model_files)}
    evaluate = {str(x.resolve()): sha(x) for x in args.evaluation_files}
    identity = dict(source=source, checkpoint=checkpoint,
                    data_manifest_sha=sha(args.data_manifest), evaluators=evaluate)
    protocol_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    versions = dict(python=platform.python_version())
    for package in ("torch", "transformers", "numpy"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = "not installed"
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        gpus = "unavailable"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(dict(protocol_id=protocol_id, identity=identity, versions=versions, gpus=gpus), f, indent=2)
    print("export E018_PROTOCOL_ID=" + protocol_id)


if __name__ == "__main__":
    main()
