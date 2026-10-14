#!/usr/bin/env python3
"""Exact-source installer. Dry check by default; --apply backs up before writing.

Newer server source is deliberately not overwritten by a guessed E014 snapshot.
Use the included scoped diff with Cursor if source hashes differ.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

BUNDLE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--require-historical-cover", action="store_true")
    p.add_argument("--rollback", type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if repo == BUNDLE:
        p.error("Target must be your complete server repository, not the delivery bundle")
    if args.rollback:
        records = json.loads((args.rollback / "installation.json").read_text())
        if records["repo"] != str(repo):
            p.error("Rollback belongs to a different repository")
        for row in records["files"]:
            if sha(repo / row["path"]) != row["installed_sha"]:
                p.error("File changed after install; refusing rollback of " + row["path"])
        if not args.apply:
            print("Rollback check OK; add --apply to restore")
            return
        for row in records["files"]:
            target = repo / row["path"]
            if row["existed"]:
                shutil.copy2(args.rollback / "files" / row["path"], target)
            else:
                target.unlink()
        print("Rollback restored " + str(repo))
        return
    manifest = json.loads((BUNDLE / "BASE_SOURCE_MANIFEST.json").read_text())
    errors = []
    for name, hashes in manifest.items():
        current = sha(repo / name)
        if current not in {hashes["before"], hashes["after"]}:
            errors.append("Source mismatch: " + name)
    deps = ["llava/model/e010_runtime_support.py", "llava/model/builder.py"]
    if args.require_historical_cover:
        deps.append("llava/model/multimodal_encoder/e013_stage1_support.py")
    for dep in deps:
        if not (repo / dep).is_file():
            errors.append("Server dependency missing: " + dep)
    all_files = [x for folder in ("llava", "scripts", "configs", "tests")
                 for x in (BUNDLE / folder).rglob("*") if x.is_file() and "__pycache__" not in x.parts]
    # Refuse to clobber independently-created E018 helpers and scripts too.
    for src in all_files:
        rel = str(src.relative_to(BUNDLE))
        if rel not in manifest and (repo / rel).exists() and sha(repo / rel) != sha(src):
            errors.append("Existing delivery file differs: " + rel)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("No files modified. Apply E014_to_E018.patch to reviewed actual server source with Cursor.", file=sys.stderr)
        raise SystemExit(2)
    print("Source and dependency checks OK (%d files)" % len(all_files))
    if not args.apply:
        print("Dry check only. Add --apply to install with a rollback backup.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = repo / "e018_backups" / stamp
    records = dict(repo=str(repo), files=[])
    # Save every existing file before any replacement.
    for src in all_files:
        rel = src.relative_to(BUNDLE)
        target = repo / rel
        existed = target.is_file()
        if existed:
            dst = backup / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dst)
        records["files"].append(dict(path=str(rel), existed=existed, installed_sha=sha(src)))
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "installation.json").write_text(json.dumps(records, indent=2))
    try:
        for src in all_files:
            target = repo / src.relative_to(BUNDLE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    except Exception:
        # Automatic rollback for a partial write; no permission detour.
        for row in records["files"]:
            target = repo / row["path"]
            if row["existed"]:
                shutil.copy2(backup / "files" / row["path"], target)
            elif target.exists():
                target.unlink()
        raise
    print("Installed; rollback backup: " + str(backup))


if __name__ == "__main__":
    main()
