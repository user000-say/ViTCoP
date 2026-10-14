#!/usr/bin/env python3
"""Report continuation headroom on held-out images. Not benchmark accuracy."""
import argparse
import json
from collections import defaultdict
import numpy as np
from e018_tools import read_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--split", default="test", choices=["train", "calibration", "test", "all"])
    p.add_argument("--lambda", dest="lam", type=float, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = [r for file in args.labels for r in read_rows(file) if args.split == "all" or r["split"] == args.split]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["fingerprint"], r["event"])].append(r)
    summaries = []
    for (fp, event), data in groups.items():
        u = np.array([r["utility"] for r in data])
        cost = np.array([r["delta_work"] for r in data])
        net = u - args.lam * cost
        summaries.append(dict(fingerprint=fp, event=event, rows=len(data),
                              images=len({str(r["image_id"]) for r in data}),
                              utility_mean=float(u.mean()), utility_median=float(np.median(u)),
                              positive_utility_fraction=float(np.mean(u > 0)),
                              harmful_continuation_fraction=float(np.mean(u < 0)),
                              local_oracle_continue_fraction=float(np.mean(net > 0)),
                              local_oracle_net_gain=float(np.maximum(net, 0).mean()),
                              now_first_correct_fraction=float(np.mean([r["now_first_correct"] for r in data])),
                              later_first_correct_fraction=float(np.mean([r["later_first_correct"] for r in data]))))
    report = dict(split=args.split, lam=args.lam, summaries=summaries,
                  limitation="Local fixed-future intervention oracle; not full policy oracle or benchmark correctness.")
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
