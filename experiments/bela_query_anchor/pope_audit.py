"""Offline POPE answer/parse audit on frozen JSONL predictions.

NO re-labeling of the official predictions: pred_label must already be
computed by the locked official evaluator. The script identifies the
FN-heavy vs FP-heavy tradeoff, per split, and paired transitions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_predictions(path):
    data = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = str(r["sample_id"])
        if key in data:
            raise ValueError(f"duplicate sample_id: {key}")
        if r.get("valid") is not True:
            raise ValueError(f"invalid/unknown answer in {path}: {key}")
        if r["label"] not in (0, 1) or r["pred_label"] not in (0, 1):
            raise ValueError(f"labels not binary in {path}: {key}")
        data[key] = r
    if not data:
        raise ValueError("no predictions")
    return data


def stats(rows):
    tp = sum(r["label"] == 1 and r["pred_label"] == 1 for r in rows)
    tn = sum(r["label"] == 0 and r["pred_label"] == 0 for r in rows)
    fp = sum(r["label"] == 0 and r["pred_label"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["pred_label"] == 0 for r in rows)
    n = tp + tn + fp + fn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return dict(n=n, TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp + tn) / n, precision=prec, recall=rec, F1=f1,
                yes_rate=(tp + fp) / n)


def grouped(data):
    all_rows = list(data.values())
    names = sorted(set(str(r.get("split", "unspecified")) for r in all_rows))
    return {"pooled": stats(all_rows), **{
        f"split:{name}": stats([r for r in all_rows if
                             str(r.get("split", "unspecified")) == name])
        for name in names
    }}


def paired(baseline, candidate):
    if set(baseline) != set(candidate):
        missing_a = sorted(set(candidate) - set(baseline))[:5]
        missing_b = sorted(set(baseline) - set(candidate))[:5]
        raise ValueError(f"sample mismatch, candidate-only {missing_a}, baseline-only {missing_b}")
    matrix = Counter()
    for key in baseline:
        a, b = baseline[key], candidate[key]
        if a["label"] != b["label"] or a.get("image_id") != b.get("image_id"):
            raise ValueError(f"label/image mismatch for {key}")
        if a.get("split") != b.get("split"):
            raise ValueError(f"split mismatch for {key}")
        matrix[(int(a["label"]), int(a["pred_label"]), int(b["pred_label"]))] += 1
    return {
        "positive_fn_to_tp": matrix[(1, 0, 1)],
        "positive_tp_to_fn": matrix[(1, 1, 0)],
        "negative_fp_to_tn": matrix[(0, 1, 0)],
        "negative_tn_to_fp": matrix[(0, 0, 1)],
        "score_delta_F1": stats(candidate.values())["F1"] - stats(baseline.values())["F1"],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    a = read_predictions(args.baseline)
    b = read_predictions(args.candidate)
    result = {"baseline": grouped(a), "candidate": grouped(b), "paired": paired(a, b)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["paired"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
