"""Build a frozen query-shuffle negative control without labels or predictions.

Inputs: JSONL rows each with sample_id and image_id.
Each sample gets a question from a different image, deterministically.
The actual question text/token IDs are loaded from the mapped sample by the
PRODUCTION adapter; this script only records IDs and makes no GPU calls.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def make_shuffle_map(rows, seed: int = 17):
    by_image = defaultdict(list)
    seen = set()
    for row in rows:
        sid, iid = str(row["sample_id"]), str(row["image_id"])
        if sid in seen:
            # Interface/identity error — not a model-quality failure. POPE's
            # question_id repeats across random/popular/adversarial; callers
            # must use a unique key (e.g. f"{row_index}|{question_id}").
            raise ValueError(
                "duplicate sample_id "
                f"{sid!r}: shuffle map requires unique sample_id per row "
                "(POPE question_id alone is not unique across subsets)"
            )
        seen.add(sid)
        by_image[iid].append(sid)
    images = sorted(by_image)
    if len(images) < 2:
        raise ValueError("shuffled-query control requires at least two different images")
    for iid in images:
        by_image[iid].sort()
    rng = random.Random(seed)
    rng.shuffle(images)
    ans = {}
    for ix, iid in enumerate(images):
        other = images[(ix + 1) % len(images)]
        donors = by_image[other]
        for j, sid in enumerate(by_image[iid]):
            ans[sid] = donors[j % len(donors)]
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    mapping = make_shuffle_map(rows, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"frozen shuffle map {len(mapping)} samples: {path}")


if __name__ == "__main__":
    main()
