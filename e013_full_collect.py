#!/usr/bin/env python3
"""Collect full-split official metrics only (never panel/limit)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e013_p1")
FULL_N = {"gqa": 12578, "pope": 9000, "mme": 2374, "mmbench_en_dev": 4329}
ARMS = [
    "BASE_S8", "B0_S8",
    "REPLACE_RANDOM16", "REPLACE_ATTN16", "REPLACE_COVER16",
    "EXTRA_RANDOM16", "EXTRA_ATTN16", "EXTRA_COVER16",
    "EXTRA_NULL16", "EXTRA_DUPLICATE16",
]


def pick(metrics: dict, task: str):
    if task == "gqa":
        return {"exact_match": metrics.get("exact_match")}
    if task == "pope":
        return {
            "pope_accuracy": metrics.get("pope_accuracy"),
            "pope_f1_score": metrics.get("pope_f1_score"),
        }
    if task == "mme":
        p = metrics.get("mme_perception_score")
        c = metrics.get("mme_cognition_score")
        total = None if p is None or c is None else float(p) + float(c)
        return {"mme_perception_score": p, "mme_cognition_score": c, "mme_total": total}
    if task == "mmbench_en_dev":
        return {"gpt_eval_score": metrics.get("gpt_eval_score")}
    return {}


def main() -> None:
    # Prefer full/full/ARM (current queue layout); also accept full/ARM.
    roots = [ROOT / "full" / "full", ROOT / "full"]
    table = {}
    missing = []
    bad_n = []
    for arm in ARMS:
        row = {"arm": arm}
        for task, need in FULL_N.items():
            path = None
            for base in roots:
                candidate = base / arm / task / "metrics.json"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                missing.append(f"{arm}/{task}")
                row[task] = None
                continue
            metrics = json.loads(path.read_text())
            n = int(metrics.get("effective") or -1)
            if n != need:
                bad_n.append(f"{arm}/{task} n={n} expected={need}")
            cell = pick(metrics, task)
            cell["effective"] = n
            cell["path"] = str(path)
            row[task] = cell
        table[arm] = row
    out = {
        "policy": "full_four_dataset_only",
        "full_n": FULL_N,
        "missing": missing,
        "bad_n": bad_n,
        "complete": not missing and not bad_n,
        "arms": table,
    }
    path = ROOT / "FULL_ACCURACY_TABLE.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"complete": out["complete"], "missing": len(missing), "bad_n": len(bad_n), "path": str(path)}))
    raise SystemExit(0 if out["complete"] else 2)


if __name__ == "__main__":
    main()
