#!/usr/bin/env python3
"""Per-arm R4 gate for NeXT-7B v4.

Prints one line starting with GO or SKIP_6ACC for the shell queue.
Rules:
  - R4 mean relative ACC must be >= --r4-min (default 92)
  - Every R4 task relative ACC must be >= --single-floor (default 85)
Relative = 100 * abs / NeXT-7B Dense (next7_dense6_absolute.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DENSE = json.loads(
    Path(
        "/home/xiangyue/ViTCoP/experiments/bela_v100_4model_qacover_20261028/results/next7_dense6_absolute.json"
    ).read_text()
)["absolute"]
DENSE_MAP = {
    "gqa": float(DENSE["gqa"]),
    "pope": float(DENSE["pope_f1"]),
    "mme": float(DENSE["mme"]),
    "mmbench_en_dev": float(DENSE["mmbench_en_dev"]),
}
R4 = ("gqa", "pope", "mme", "mmbench_en_dev")


def extract_abs(task: str, results: dict) -> float | None:
    block = results.get("results", {}).get(task, {})
    if not isinstance(block, dict):
        return None
    if task == "gqa":
        v = block.get("exact_match,none")
        return None if v is None else 100.0 * float(v)
    if task == "pope":
        for k in ("pope_f1_score,none", "f1_score,none"):
            if k in block:
                return 100.0 * float(block[k])
        return None
    if task == "mme":
        p, c = block.get("mme_perception_score,none"), block.get("mme_cognition_score,none")
        if p is None or c is None:
            return None
        return float(p) + float(c)
    if task == "mmbench_en_dev":
        for k in ("gpt_eval_score,none", "gpt_eval_acc,none", "accuracy,none"):
            if k in block:
                return float(block[k])
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--r4-min", type=float, default=92.0)
    ap.add_argument("--single-floor", type=float, default=85.0)
    args = ap.parse_args()
    root = Path(args.out_root)
    abs_s: dict[str, float] = {}
    rel_s: dict[str, float] = {}
    missing = []
    for t in R4:
        hits = sorted(
            p
            for p in (root / args.arm / t).glob("**/*_results.json")
            if "submission" not in str(p)
        )
        if not hits:
            missing.append(t)
            continue
        a = extract_abs(t, json.loads(hits[0].read_text()))
        if a is None:
            missing.append(t)
            continue
        abs_s[t] = a
        rel_s[t] = 100.0 * a / DENSE_MAP[t]

    payload = {
        "arm": args.arm,
        "r4_min": args.r4_min,
        "single_floor": args.single_floor,
        "abs": abs_s,
        "rel": rel_s,
        "missing": missing,
        "dense_ref": DENSE_MAP,
    }
    if missing:
        payload["verdict"] = "SKIP_6ACC"
        payload["reason"] = f"incomplete_r4:{missing}"
        print(f"SKIP_6ACC incomplete_r4 missing={missing}")
    else:
        r4 = sum(rel_s[t] for t in R4) / 4.0
        weak = {t: rel_s[t] for t in R4 if rel_s[t] < args.single_floor}
        payload["r4_mean_rel"] = r4
        payload["weak"] = weak
        if weak:
            payload["verdict"] = "SKIP_6ACC"
            payload["reason"] = f"single_below_{args.single_floor}"
            print(
                f"SKIP_6ACC single_below_{args.single_floor} weak={ {k: round(v,2) for k,v in weak.items()} } r4={r4:.2f}"
            )
        elif r4 < args.r4_min:
            payload["verdict"] = "SKIP_6ACC"
            payload["reason"] = f"r4_below_{args.r4_min}"
            print(f"SKIP_6ACC r4_below_{args.r4_min} r4={r4:.2f} rel={ {k: round(v,2) for k,v in rel_s.items()} }")
        else:
            payload["verdict"] = "GO"
            payload["reason"] = "pass"
            print(f"GO r4={r4:.2f} rel={ {k: round(v,2) for k,v in rel_s.items()} }")

    gate_path = root / args.arm / "R4_GATE.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
