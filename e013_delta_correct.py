#!/usr/bin/env python3
"""Offline Δcorrect analysis from full-split log_samples (addendum utility stage-1).

No gold log-prob. Paired by sample_id within each task. Image-cluster bootstrap
when image_id is present in samples. Never treats panel/limit as ACC.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e013_p1/full/full")
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")


def load_samples(arm: str, task: str) -> dict[str, dict]:
    base = ROOT / arm / task
    files = [p for p in base.rglob("*.jsonl") if "samples" in p.name or "log" in p.name.lower()]
    # lmms-eval typically writes *_samples_*.jsonl next to results
    files += list(base.rglob("*_samples_*.jsonl"))
    files = sorted(set(files))
    out = {}
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("doc_id", row.get("sample_id", row.get("id", ""))))
            # correctness heuristics used by lmms-eval dumps
            correct = row.get("exact_match")
            if correct is None:
                filtered = row.get("filtered_resps") or row.get("resps")
                target = row.get("target") or row.get("answer")
                if isinstance(filtered, list) and filtered and target is not None:
                    pred = filtered[0][0] if isinstance(filtered[0], list) else filtered[0]
                    correct = float(str(pred).strip().lower() == str(target).strip().lower())
            if correct is None:
                continue
            img = row.get("image_id") or row.get("doc", {}).get("imageId") if isinstance(row.get("doc"), dict) else row.get("image_id")
            out[sid] = {"correct": float(correct), "image_id": img, "path": str(path)}
    return out


def paired(base: dict, other: dict):
    keys = sorted(set(base) & set(other))
    rows = []
    for k in keys:
        b, o = base[k]["correct"], other[k]["correct"]
        rows.append({
            "sample_id": k,
            "image_id": base[k].get("image_id") or other[k].get("image_id"),
            "correct_base": b,
            "correct_other": o,
            "delta": o - b,
        })
    return rows


def bootstrap_mean(deltas, image_ids, n=1000, seed=20261009):
    rng = random.Random(seed)
    # cluster by image when available
    clusters = {}
    for d, img, i in zip(deltas, image_ids, range(len(deltas))):
        key = img if img is not None else f"sid:{i}"
        clusters.setdefault(key, []).append(d)
    keys = list(clusters)
    means = []
    for _ in range(n):
        draw = [rng.choice(keys) for _ in keys]
        vals = [v for k in draw for v in clusters[k]]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    means.sort()
    return {
        "mean": sum(deltas) / len(deltas) if deltas else None,
        "ci95": [means[int(0.025 * n)], means[int(0.975 * n)]] if means else None,
        "n": len(deltas),
        "n_clusters": len(keys),
        "benefit_rate": sum(1 for d in deltas if d > 0) / len(deltas) if deltas else None,
        "harm_rate": sum(1 for d in deltas if d < 0) / len(deltas) if deltas else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BASE_S8")
    ap.add_argument("--others", nargs="+", default=[
        "REPLACE_COVER16", "REPLACE_ATTN16", "REPLACE_RANDOM16",
        "EXTRA_COVER16", "EXTRA_ATTN16", "EXTRA_RANDOM16",
        "EXTRA_NULL16", "EXTRA_DUPLICATE16",
    ])
    ap.add_argument("--out", default="/home/xiangyue/ViTCoP/logs/e013_p1/DELTA_CORRECT.json")
    args = ap.parse_args()
    report = {"base": args.base, "tasks": {}}
    for task in TASKS:
        base = load_samples(args.base, task)
        report["tasks"][task] = {}
        for arm in args.others:
            other = load_samples(arm, task)
            rows = paired(base, other)
            if not rows:
                report["tasks"][task][arm] = {"status": "NO_PAIRED_SAMPLES"}
                continue
            stats = bootstrap_mean(
                [r["delta"] for r in rows],
                [r["image_id"] for r in rows],
            )
            report["tasks"][task][arm] = {"status": "OK", **stats}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": args.out, "summary": {
        t: {a: v.get("status") for a, v in arms.items()} for t, arms in report["tasks"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
