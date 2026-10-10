#!/usr/bin/env python3
"""Collect E014 full-split metrics into experiments/e014_aee tables."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e014_aee"
OUT = REPO / "experiments/e014_aee"
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
EXPECTED = {"gqa": 12578, "pope": 9000, "mme": 2374, "mmbench_en_dev": 4329}


def score_of(task: str, metrics: dict):
    if task == "gqa":
        return metrics.get("exact_match") or metrics.get("accuracy") or metrics.get("gqa")
    if task == "pope":
        return metrics.get("pope_f1_score") or metrics.get("f1") or metrics.get("F1")
    if task == "mme":
        return metrics.get("mme_total") or metrics.get("mme") or metrics.get("total")
    if task == "mmbench_en_dev":
        return metrics.get("mmbench_en_dev") or metrics.get("accuracy") or metrics.get("score")
    return None


def n_of(metrics: dict):
    for key in ("effective", "n", "num_examples", "count", "samples"):
        if key in metrics and metrics[key] is not None:
            try:
                return int(metrics[key])
            except (TypeError, ValueError):
                pass
    return None


def collect(phase: str) -> dict:
    base = ROOT / "full" / phase
    rows = []
    if not base.exists():
        return {"phase": phase, "rows": [], "complete": False}
    for arm_dir in sorted(base.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        row = {"arm": arm, "phase": phase}
        ok = True
        for task in TASKS:
            path = arm_dir / task / "metrics.json"
            if not path.exists():
                row[task] = None
                row[f"{task}_n"] = None
                row[f"{task}_status"] = "MISSING"
                ok = False
                continue
            metrics = json.loads(path.read_text())
            n = n_of(metrics)
            score = score_of(task, metrics)
            # nested lmms style
            if score is None and isinstance(metrics.get("results"), dict):
                score = score_of(task, metrics["results"])
                n = n or n_of(metrics["results"])
            exp = EXPECTED[task]
            status = "VERIFIED_FULL_N" if n == exp else ("FAILED_FULL_N" if n else "UNKNOWN_N")
            if status != "VERIFIED_FULL_N":
                ok = False
            row[task] = score
            row[f"{task}_n"] = n
            row[f"{task}_status"] = status
        row["arm_ok"] = ok
        rows.append(row)
    return {
        "phase": phase,
        "rows": rows,
        "complete": bool(rows) and all(r.get("arm_ok") for r in rows),
        "expected_n": EXPECTED,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for phase in ("full_k96", "full_k64", "full_k32"):
        summary[phase] = collect(phase)
        path = OUT / f"{phase}_ACCURACY_TABLE.json"
        path.write_text(json.dumps(summary[phase], indent=2), encoding="utf-8")
        print(phase, "arms", len(summary[phase]["rows"]), "complete", summary[phase]["complete"])
    (OUT / "FULL_ACCURACY_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
