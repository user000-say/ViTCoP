"""Paired, image-clustered analysis of official per-item score exports.

This is not an answer parser or a replacement for lmms-eval. MME composite
scores require the official evaluator and are deliberately not averaged here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

MEAN_TASKS = {"gqa", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr"}


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty prediction export")
    return rows


def align(a, b):
    def index(rows):
        result = {}
        for r in rows:
            key = (str(r["task"]), str(r["sample_id"]))
            if key in result:
                raise ValueError("duplicate sample ID")
            result[key] = r
        return result
    aa, bb = index(a), index(b)
    if aa.keys() != bb.keys():
        raise ValueError("sample sets differ; never silently analyze the intersection")
    keys = sorted(aa)
    for k in keys:
        if str(aa[k]["image_id"]) != str(bb[k]["image_id"]):
            raise ValueError("paired image IDs differ")
        if k[0] == "pope" and (aa[k].get("label") != bb[k].get("label")
                                  or aa[k].get("split") != bb[k].get("split")):
            raise ValueError("paired POPE labels/splits differ")
    return [aa[k] for k in keys], [bb[k] for k in keys]


def confusion(rows):
    out = dict(tp=0, fp=0, tn=0, fn=0, invalid=0, n=len(rows))
    for r in rows:
        y, pred = r.get("label"), r.get("pred_label")
        if y not in (0, 1) or pred not in (0, 1):
            raise ValueError("POPE requires official-parser binary labels; null is not 'no'")
        key = {(1, 1): "tp", (0, 1): "fp", (0, 0): "tn", (1, 0): "fn"}[(y, pred)]
        out[key] += 1
        out["invalid"] += int(not r.get("valid", True))
    tp, fp, fn = out["tp"], out["fp"], out["fn"]
    out["f1"] = 100 * (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    out["precision"] = 100 * tp / (tp + fp) if tp + fp else 0.0
    out["recall"] = 100 * tp / (tp + fn) if tp + fn else 0.0
    out["yes_rate"] = 100 * (tp + fp) / len(rows) if rows else 0.0
    return out


def metric(rows, task, pope_reduction=None, splits=None):
    if task in MEAN_TASKS:
        scores = np.array([r["score"] for r in rows], dtype=float)
        if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("official scores must be finite in [0,1]")
        return float(scores.mean() * 100)
    if task == "pope":
        if pope_reduction == "pooled":
            return confusion(rows)["f1"]
        if pope_reduction == "macro":
            names = splits or sorted({str(r["split"]) for r in rows})
            if not names:
                raise ValueError("missing POPE splits")
            return float(np.mean([confusion([r for r in rows if str(r["split"]) == s])["f1"]
                                  for s in names]))
        raise ValueError("explicit POPE pooled/macro reduction required")
    raise ValueError("use the official composite scorer for MME; no generic row average")


def paired_report(a, b, task, reps=2000, seed=17, pope_reduction=None):
    a, b = align(a, b)
    if {str(r["task"]) for r in a} != {task}:
        raise ValueError("analyze one task per report")
    splits = sorted({str(r.get("split")) for r in a}) if task == "pope" else None
    sa, sb = metric(a, task, pope_reduction, splits), metric(b, task, pope_reduction, splits)
    groups = {}
    for i, r in enumerate(a):
        groups.setdefault(str(r["image_id"]), []).append(i)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        chosen = rng.choice(group_ids, len(group_ids), replace=True)
        idx = [i for g in chosen for i in groups[str(g)]]
        aa, bb = [a[i] for i in idx], [b[i] for i in idx]
        deltas.append(metric(bb, task, pope_reduction, splits) - metric(aa, task, pope_reduction, splits))
    if task == "pope":
        item_a = np.array([r["label"] == r["pred_label"] for r in a], dtype=float)
        item_b = np.array([r["label"] == r["pred_label"] for r in b], dtype=float)
    else:
        item_a = np.array([r["score"] for r in a], dtype=float)
        item_b = np.array([r["score"] for r in b], dtype=float)
    out = {"task": task, "n_items": len(a), "n_images": len(groups),
           "baseline": sa, "candidate": sb, "delta": sb - sa,
           "cluster_bootstrap_95ci": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
           "candidate_improved_items": int((item_b > item_a).sum()),
           "candidate_worsened_items": int((item_b < item_a).sum()),
           "pope_reduction": pope_reduction if task == "pope" else None,
           "note": "Pilot CIs can be wide. An inconclusive pilot is not a demonstrated null effect."}
    if task == "pope":
        out["baseline_confusion"] = confusion(a)
        out["candidate_confusion"] = confusion(b)
        out["by_split"] = {s: {"baseline": confusion([r for r in a if str(r["split"]) == s]),
                                "candidate": confusion([r for r in b if str(r["split"]) == s])}
                           for s in splits}
    elif set(np.unique(np.r_[item_a, item_b])).issubset({0.0, 1.0}):
        out["binary_transitions"] = {
            "both_correct": int(((item_a == 1) & (item_b == 1)).sum()),
            "baseline_only": int(((item_a == 1) & (item_b == 0)).sum()),
            "candidate_only": int(((item_a == 0) & (item_b == 1)).sum()),
            "both_wrong": int(((item_a == 0) & (item_b == 0)).sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--pope-reduction", choices=["pooled", "macro"])
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.reps < 0:
        p.error("reps must be nonnegative")
    out = paired_report(load(args.baseline), load(args.candidate), args.task,
                        args.reps, args.seed, args.pope_reduction)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
