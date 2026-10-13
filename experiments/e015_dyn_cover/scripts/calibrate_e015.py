#!/usr/bin/env python3
"""Calibrate E015 exchange-budget thresholds using *separate unlabeled images*.

Input: logs from E014_VARIANT=E015_PROBE,E014_AUDIT=True, tagged
[E014_STAGE1_JSON] with coverage_gap. Never use held-out evaluation answers.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

MARKER = "[E014_STAGE1_JSON] "

def percentile(sorted_values, p):
    n = len(sorted_values)
    x = p * (n - 1)
    lo, hi = int(math.floor(x)), int(math.ceil(x))
    return sorted_values[lo] + (x - lo) * (sorted_values[hi] - sorted_values[lo])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("e015_thresholds.json"))
    ap.add_argument("--min-images", type=int, default=128)
    args=ap.parse_args()
    records=[]
    for p in args.logs:
        for line in p.read_text(errors="replace").splitlines():
            if MARKER not in line:
                continue
            row=json.loads(line.split(MARKER,1)[1])
            if row.get("variant") != "E015_PROBE":
                continue
            v=float(row["coverage_gap"])
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("nonfinite/out-of-range coverage gap")
            records.append(v)
    if len(records)<args.min_images:
        raise ValueError(f"need >= {args.min_images} independent calibration images, got {len(records)}")
    records.sort()
    q=[percentile(records,p) for p in (.15,.35,.65,.85)]
    spread = q[-1] - q[0]
    if len(set(q))<4 or spread < 1e-5:
        raise ValueError("coverage-gap collapsed: do not claim image-adaptive routing")
    result={
        "n_calibration_images":len(records),
        "gap_mean":statistics.mean(records),
        "gap_median":statistics.median(records),
        "gap_min":records[0],
        "gap_max":records[-1],
        "percentiles":[.15,.35,.65,.85],
        "thresholds":q,
        "E015_THRESHOLDS":",".join(f"{x:.10g}" for x in q),
        "warning":"separate unlabeled image calibration; predictions and evaluation labels forbidden"
    }
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("export E015_THRESHOLDS='"+result["E015_THRESHOLDS"]+"'")
    print("calibration summary",result)

if __name__=="__main__":
    main()
