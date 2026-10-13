#!/usr/bin/env python3
"""Verifiable macro R4 with explicitly supplied SAME-PROTOCOL baseline.

Never substitute a paper's "Rel." or 10-task aggregate for this R4.
"""
import argparse, csv, math
from pathlib import Path

FIELDS=("gqa","pope","mme","mmbench")
E014_BASELINE=(61.9,85.8,1852.0,64.08)

def r4(raw, vanilla):
    a=[float(i) for i in raw]
    b=[float(i) for i in vanilla]
    if len(a)!=4 or len(b)!=4 or any(not math.isfinite(z) for z in a+b):
        raise ValueError("require four finite source metrics + baselines")
    if any(z<=0 for z in b):
        raise ValueError("baseline metrics must be positive")
    ratios=[100*a[i]/b[i] for i in range(4)]
    return {"per_metric_retention":dict(zip(FIELDS,ratios)),
            "r4":sum(ratios)/4.0}

def vector(s):
    v=[float(x.strip()) for x in s.split(",")]
    if len(v)!=4: raise ValueError("supply four scores in order GQA, POPE, MME, MMBench")
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",help="gqa,pope,mme,mmbench in SAME eval protocol")
    ap.add_argument("--baseline",
        help="baseline gqa,pope,mme,mmbench in EXACT same protocol")
    ap.add_argument("--e014-baseline",action="store_true",
        help="use frozen E014 LLaVA7B metrics [61.9,85.8,1852,64.08]")
    ap.add_argument("--csv",type=Path,help="input CSV with columns method,gqa,pope,mme,mmbench")
    args=ap.parse_args()
    if bool(args.baseline)==bool(args.e014_baseline):
        ap.error("choose --baseline OR --e014-baseline")
    ref=vector(args.baseline) if args.baseline else E014_BASELINE
    def emit(label,a):
        z=r4(a,ref)
        print(f"{label}: R4={z['r4']:.6f}% with baseline={ref}")
        for field,v in z["per_metric_retention"].items():
            print(f"  {field}={v:.6f}%")
    if args.csv:
        with args.csv.open(newline="") as f:
            rows=csv.DictReader(f)
            for row in rows: emit(row.get("method","unnamed"),[row[k] for k in FIELDS])
    elif args.raw:
        emit("single",vector(args.raw))
    else: ap.error("choose --raw or --csv")

if __name__=="__main__":
    main()
