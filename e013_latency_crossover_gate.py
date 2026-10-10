#!/usr/bin/env python3
"""ABBA timing gate: LEGACY L12 vs E012 S12D12 on Fixed32.

User / project rule (not the vague handoff 'band'): relative |Δ| Fixed32
mean must stay within max_rel_percent (default 15%) of the LEGACY baseline.
ABBA failing blocks attributing Stage1 wall-clock gains to the algorithm;
accuracy results remain valid either way.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> dict[str, list[float]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by: dict[str, list[float]] = {}
    for row in rows:
        by.setdefault(row["sample_id"], []).append(float(row["fixed32_ms"]))
    return by


def per_sample_means(data: dict[str, list[float]]) -> dict[str, float]:
    out = {}
    for key, vals in data.items():
        if len(vals) != 5:
            raise SystemExit(f"repeats {key} got {len(vals)}")
        out[key] = statistics.mean(vals)
    return out


def paired_rel(legacy: dict[str, float], other: dict[str, float]) -> list[float]:
    """(other - legacy) / legacy per sample. Positive => other slower."""
    keys = sorted(set(legacy) & set(other))
    if len(keys) != 128:
        raise SystemExit(f"paired samples {len(keys)} expected 128")
    out = []
    for key in keys:
        base = legacy[key]
        if base <= 0:
            raise SystemExit(f"nonpositive baseline {key}={base}")
        out.append((other[key] - base) / base)
    return out


def paired_diff(a: dict[str, float], b: dict[str, float]) -> list[float]:
    keys = sorted(set(a) & set(b))
    if len(keys) != 128:
        raise SystemExit(f"paired samples {len(keys)} expected 128")
    return [a[k] - b[k] for k in keys]


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-a", required=True)
    ap.add_argument("--e012-a", required=True)
    ap.add_argument("--e012-b", required=True)
    ap.add_argument("--legacy-b", required=True)
    ap.add_argument(
        "--max-rel-percent",
        type=float,
        default=15.0,
        help="accept if |mean relative Fixed32 delta| <= this percent (user standard)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    la = per_sample_means(load(Path(args.legacy_a)))
    ea = per_sample_means(load(Path(args.e012_a)))
    eb = per_sample_means(load(Path(args.e012_b)))
    lb = per_sample_means(load(Path(args.legacy_b)))

    # Pair1: E012_A vs LEGACY_A; Pair2: E012_B vs LEGACY_B (same ABBA order science).
    r1 = paired_rel(la, ea)
    r2 = paired_rel(lb, eb)
    # Stability: same method A vs B relative to first run.
    stab_l = paired_rel(la, lb)
    stab_e = paired_rel(ea, eb)
    # Absolute ms diffs kept for logging only.
    d1 = paired_diff(la, ea)
    d2 = paired_diff(lb, eb)

    m1, lo1, hi1 = ci95(r1)
    m2, lo2, hi2 = ci95(r2)
    ms, los, his = ci95(stab_l)
    me, loe, hie = ci95(stab_e)
    md1, _, _ = ci95(d1)
    md2, _, _ = ci95(d2)

    limit = args.max_rel_percent / 100.0
    ok = abs(m1) <= limit and abs(m2) <= limit
    report = {
        "status": "PASS" if ok else "FAIL",
        "rule": "user_relative_15pct_fixed32",
        "max_rel_percent": args.max_rel_percent,
        "e012_minus_legacy_pair1_rel_mean_ci": [m1, lo1, hi1],
        "e012_minus_legacy_pair2_rel_mean_ci": [m2, lo2, hi2],
        "e012_minus_legacy_pair1_ms_mean": md1,
        "e012_minus_legacy_pair2_ms_mean": md2,
        "legacy_ab_stability_rel_mean_ci": [ms, los, his],
        "e012_ab_stability_rel_mean_ci": [me, loe, hie],
        "legacy_a_mean_fixed32_ms": statistics.mean(la.values()),
        "e012_a_mean_fixed32_ms": statistics.mean(ea.values()),
        "verdict": (
            f"|rel Fixed32 mean| <= {args.max_rel_percent}% on both ABBA pairs; "
            "Stage1 latency attribution allowed"
            if ok else
            f"|rel Fixed32 mean| exceeded {args.max_rel_percent}% on an ABBA pair; "
            "do not attribute E012/E013 speed gains to algorithm (accuracy still valid)"
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
