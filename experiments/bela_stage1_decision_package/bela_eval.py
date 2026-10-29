#!/usr/bin/env python3
"""BELA research diagnostics: paired predictions + Stage1 trace audit.
Uses only exported model predictions and trace records.
No GPU experiment is performed by this script.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            items.append(value)
    if not items:
        raise ValueError(f"empty file: {path}")
    return items


def unique_index(rows):
    index = {}
    for row in rows:
        sid = str(row.get("sample_id", ""))
        if not sid:
            raise ValueError("every row needs a stable sample_id")
        if sid in index:
            raise ValueError(f"duplicate sample_id {sid!r}; POPE question_id is NOT a unique key")
        index[sid] = row
    return index


def bool_correct(row):
    if "correct" in row:
        value = row["correct"]
        if type(value) is bool:
            return int(value)
        if type(value) is int and value in (0, 1):
            return value
        raise ValueError("'correct' must be a boolean or 0/1")
    if "gold" not in row or "pred" not in row:
        raise ValueError("GQA row needs official 'correct' or gold/pred")
    return int(str(row["gold"]).strip().casefold() == str(row["pred"]).strip().casefold())


def cm_row(row):
    gold = str(row.get("gold", "")).strip().lower()
    pred = str(row.get("pred", "")).strip().lower()
    if gold not in ("yes", "no") or pred not in ("yes", "no"):
        raise ValueError(f"POPE gold/pred must be yes/no: {row.get('sample_id')}")
    return np.array([int(gold=="yes" and pred=="yes"),
                     int(gold=="no" and pred=="yes"),
                     int(gold=="yes" and pred=="no"),
                     int(gold=="no" and pred=="no")], dtype=np.int64)


def metric_from_totals(totals, metric):
    totals = np.asarray(totals)
    if metric == "gqa":
        return 100.0 * totals[..., 0] / np.maximum(totals[..., 1], 1)
    tp, fp, fn = [totals[..., i] for i in range(3)]
    return 100.0 * 2*tp / np.maximum(2*tp+fp+fn, 1)


def summary(rows, metric):
    if metric == "gqa":
        correct = sum(bool_correct(x) for x in rows)
        exact_score = metric_from_totals([correct, len(rows)], metric)
        return {"accuracy": float(exact_score), "correct": correct, "n":len(rows),
                "scoring_basis": "official_correct" if all("correct" in x for x in rows)
                else "simple_EM_not_official"}
    cm = np.stack([cm_row(x) for x in rows]).sum(axis=0)
    tp, fp, fn, tn = [int(x) for x in cm]
    n = tp+fp+fn+tn
    return {"f1": float(metric_from_totals(cm, "pope")),
            "accuracy": 100*(tp+tn)/n, "precision": 100*tp/max(tp+fp,1),
            "recall": 100*tp/max(tp+fn,1),
            "yes_rate": 100*(tp+fp)/n,
            "tp":tp, "fp":fp, "fn":fn, "tn":tn, "n":n}


def paired_compare(baseline_rows, candidate_rows, metric, n_boot=1000, seed=27):
    before = unique_index(baseline_rows)
    after = unique_index(candidate_rows)
    if set(before) != set(after):
        only_a = len(set(before)-set(after))
        only_b = len(set(after)-set(before))
        raise ValueError(f"sample_id sets differ: only_baseline={only_a}, only_candidate={only_b}")
    if metric not in ("gqa", "pope"):
        raise ValueError("metric must be 'gqa' or 'pope'")
    ids = sorted(before)
    a, b = [before[k] for k in ids], [after[k] for k in ids]
    for ra, rb in zip(a,b):
        if str(ra.get("image_id","")) != str(rb.get("image_id","")):
            raise ValueError(f"image_id mismatch for {ra['sample_id']}")
        if "gold" in ra and "gold" in rb and str(ra["gold"]) != str(rb["gold"]):
            raise ValueError(f"gold mismatch for {ra['sample_id']}")
        if metric == "gqa" and ("correct" in ra) != ("correct" in rb):
            raise ValueError("both GQA arms must use the same scoring basis")
    groups = defaultdict(list)
    for i, ra in enumerate(a):
        # Cluster by image to avoid treating questions from the same image as independent.
        g = str(ra.get("image_id") or ra["sample_id"])
        groups[g].append(i)
    keys = sorted(groups)
    if metric == "gqa":
        mat_a = np.array([[bool_correct(x), 1] for x in a], dtype=np.int64)
        mat_b = np.array([[bool_correct(x), 1] for x in b], dtype=np.int64)
        flips = Counter(f"{bool_correct(ra)}->{bool_correct(rb)}" for ra,rb in zip(a,b))
    else:
        mat_a = np.stack([cm_row(x) for x in a])
        mat_b = np.stack([cm_row(x) for x in b])
        flips = Counter(f"{ra['pred'].strip().lower()}->{rb['pred'].strip().lower()}"
                        for ra,rb in zip(a,b))
    group_a = np.stack([mat_a[groups[k]].sum(axis=0) for k in keys])
    group_b = np.stack([mat_b[groups[k]].sum(axis=0) for k in keys])
    rng = np.random.default_rng(seed)
    n_groups = len(keys)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.integers(0, n_groups, size=n_groups)
        sa, sb = group_a[sample].sum(axis=0), group_b[sample].sum(axis=0)
        deltas[i] = metric_from_totals(sb, metric) - metric_from_totals(sa, metric)
    score_a, score_b = summary(a, metric), summary(b, metric)
    primary_key = "accuracy" if metric == "gqa" else "f1"
    return {
        "metric": metric, "n":len(ids), "independent_image_clusters": n_groups,
        "baseline":score_a, "candidate":score_b,
        "delta_pp": score_b[primary_key]-score_a[primary_key],
        "paired_cluster_bootstrap_ci95_pp":[float(x) for x in np.percentile(deltas,[2.5,97.5])],
        "flips":dict(flips), "bootstrap_repetitions":n_boot, "seed":seed,
        "warning":"Historical GQA/POPE test-set selection invalidates confirmatory p-values"
    }


def check_ids(ids, n, label):
    if not isinstance(ids, list) or any(type(x) is not int for x in ids):
        raise ValueError(f"{label} must be a list of integer patch IDs")
    if len(set(ids)) != len(ids) or any(x<0 or x>=n for x in ids):
        raise ValueError(f"{label} has duplicate or out-of-range patch IDs")
    return ids


def audit_trace(rows):
    swaps, net, overlaps, at_cap, returned = [], [], [], 0, 0
    invalid = []
    for i, row in enumerate(rows):
        sid = row.get("sample_id", str(i))
        try:
            n = int(row["n_patches"])
            if n<1: raise ValueError("n_patches must be >=1")
            orig = check_ids(row["native_ids"], n, "native_ids")
            final = check_ids(row["final_ids"], n, "final_ids")
            if len(orig)!=len(final):
                raise ValueError("entry K changed")
            cap = int(row["max_swaps"])
            cnt = int(row["swaps"])
            if cap<0 or cnt<0 or cnt>min(cap,len(orig),n-len(orig)):
                raise ValueError("invalid number of accepted swaps")
            actions = row.get("records")
            if actions is not None:
                if len(actions)!=cnt:
                    raise ValueError("number of actions doesn't match swaps")
                active = set(orig)
                for x in actions:
                    r, a = int(x["removed_patch"]),int(x["added_patch"])
                    if r not in active or a in active or a<0 or a>=n:
                        raise ValueError(f"invalid swap action ({r}, {a})")
                    active.remove(r); active.add(a)
                    if "coverage_delta" in x and "max_coverage_loss" in row:
                        if float(x["coverage_delta"]) < -float(row["max_coverage_loss"])-1e-7:
                            raise ValueError("coverage guard violated")
                    if "query_gain" in x and "min_query_gain" in row:
                        if float(x["query_gain"]) < float(row["min_query_gain"])-1e-7:
                            raise ValueError("query gain guard violated")
                if active!=set(final):
                    raise ValueError("replayed swaps disagree with final_ids")
            swaps.append(cnt)
            net.append(len(set(final)-set(orig)))
            overlaps.append(len(set(final)&set(orig))/max(len(orig),1))
            at_cap += int(cnt==min(cap,len(orig),n-len(orig)) and cap>0)
            returned += int(len(set(final)-set(orig))<cnt)
        except (ValueError, TypeError, KeyError) as exc:
            invalid.append({"sample_id":sid,"error":str(exc)})
    if invalid:
        raise ValueError("trace audit failed: "+json.dumps(invalid[:10], ensure_ascii=False))
    return {
        "n_samples":len(rows),
        "mean_accepted_swaps":float(np.mean(swaps)),
        "fraction_at_hard_cap":at_cap/len(rows),
        "mean_net_new_patch_ids":float(np.mean(net)),
        "mean_original_overlap":float(np.mean(overlaps)),
        "samples_with_reentry_or_cancellation":returned,
        "mean_effective_swap_fraction":float(np.mean(
            [x/y if y else 0 for x,y in zip(net,swaps)]))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("compare")
    p.add_argument("--baseline",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--metric",required=True, choices=["gqa","pope"])
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--seed",type=int,default=27)
    q=sub.add_parser("audit")
    q.add_argument("--trace",required=True)
    p.add_argument("--out", required=True)
    q.add_argument("--out", required=True)
    args=parser.parse_args()
    if args.mode=="compare":
        report=paired_compare(read_jsonl(args.baseline),read_jsonl(args.candidate),
                              args.metric,args.bootstrap,args.seed)
    else:
        report=audit_trace(read_jsonl(args.trace))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                              encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
