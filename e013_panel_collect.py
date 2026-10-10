#!/usr/bin/env python3
"""Collect E013 panel metrics into a summary table (exact_match / F1 / MME / MMBench)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/logs/e013_p1")


def score(metrics: dict, task: str) -> float | None:
    if task == "gqa":
        return metrics.get("exact_match")
    if task == "pope":
        return metrics.get("pope_f1_score")
    if task == "mme":
        p, c = metrics.get("mme_perception_score"), metrics.get("mme_cognition_score")
        if p is None or c is None:
            return None
        return float(p) + float(c)
    if task == "mmbench_en_dev":
        return metrics.get("gpt_eval_score")
    return None


def collect(phase: str) -> dict:
    base = ROOT / "panel" / phase
    rows = {}
    for arm_dir in sorted(base.glob("*")):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        row = {"arm": arm}
        for task_dir in sorted(arm_dir.glob("*")):
            path = task_dir / "metrics.json"
            if not path.exists():
                row[task_dir.name] = None
                continue
            metrics = json.loads(path.read_text())
            row[task_dir.name] = score(metrics, task_dir.name)
            row[f"{task_dir.name}_n"] = metrics.get("effective")
        rows[arm] = row
    return rows


def main() -> None:
    out = {}
    for phase in ("screen1", "screen2", "bridge12"):
        if (ROOT / "panel" / phase).exists():
            out[phase] = collect(phase)
    path = ROOT / "PANEL_SUMMARY.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
