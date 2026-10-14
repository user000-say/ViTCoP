#!/usr/bin/env python3
"""E014 diagnostics: analyze offline Stage1 audits and full-N evaluation manifest.

Use separate E014_AUDIT=True runs; publication latency runs keep auditing OFF.
Scores are incomparable across benchmarks without normalization; the macro
column is the mean of relative changes against same-K BASE, NOT raw accuracy.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

def audit_records(paths):
    for name in paths:
        with Path(name).open(encoding="utf-8",errors="replace") as fp:
            for line in fp:
                prefix = "[E014_STAGE1_JSON] "
                if prefix in line:
                    yield json.loads(line.split(prefix,1)[1])

def audit_summary(paths):
    grouped = defaultdict(list)
    for rec in audit_records(paths):
        grouped[rec["variant"]].append(rec)
    for method, rows in grouped.items():
        counts=Counter(int(x["budget"]) for x in rows)
        print(f"Audit: {method}, n={len(rows)}, b_hist={dict(sorted(counts.items()))}, "
              f"n_distinct_b={len(counts)}")
        if len(counts)==1:
            print(" WARNING: budget is degenerate on this sample")
        for row in rows:
            if (len(row["selected_ids"]) != 172 or
                len(set(row["selected_ids"])) != 172 or
                len(row["added_ids"]) != row["budget"] or
                len(row["removed_ids"]) != row["budget"]):
                raise RuntimeError("Malformed E014 Stage1 audit")

def full_report(path):
    records=[json.loads(s) for s in Path(path).read_text(
        encoding="utf-8").splitlines() if s.strip()]
    verified=[x for x in records if x.get("status")=="VERIFIED_FULL_N"]
    print("Full-N verified:",len(verified),"/",len(records))
    if len(verified)!=len(records):
        print(" WARNING: NOT ALL full runs have valid counts; exclude failed jobs")
    grouped=defaultdict(dict)
    for r in verified:
        s=r.get("summary",{})
        if "score" in s:
            grouped[(r["k"],r["method"],r["b"])][r["benchmark"]]=float(s["score"])
    by_k=defaultdict(list)
    for (k,method,b), metric in grouped.items():
        if len(metric)!=4:
            continue
        baseline=grouped.get((k,"BASE",0),{})
        if len(baseline)!=4:
            continue
        # Mean of four within-task relative (%) changes vs same-K BASE.
        # It is NOT an accuracy percentage-point difference.
        relative_pct=sum((metric[d]/baseline[d]-1)*100 for d in baseline)/4
        by_k[k].append((relative_pct,method,b,metric))
    for k,points in sorted(by_k.items()):
        print(f"Final K={k}, four-dataset normalized macro vs same-K BASE:")
        for rel,method,b,metrics in sorted(points,reverse=True):
            print(f"  {method:20s} B={b:2d}  macro_rel_change_pct={rel:+.4f}"
                  f"  GQA={metrics['GQA']:.4f} POPE={metrics['POPE']:.4f}"
                  f" MME={metrics['MME']:.4f} MMBench={metrics['MMBench']:.4f}")
    if not by_k:
        print(" No complete 4-benchmark scored rows to compare yet. "
              "Every eval_summary.json needs {n:<int>,score:<float>}.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit-log",nargs="*",default=[])
    p.add_argument("--manifest")
    args=p.parse_args()
    if args.audit_log:
        audit_summary(args.audit_log)
    if args.manifest:
        full_report(args.manifest)
    if not args.audit_log and not args.manifest:
        p.error("Provide --audit-log and/or --manifest")

if __name__=="__main__":
    main()
