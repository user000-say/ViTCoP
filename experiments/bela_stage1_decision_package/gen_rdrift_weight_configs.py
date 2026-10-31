#!/usr/bin/env python3
"""Emit RDRIFT weight × compression configs for the winning ABC schedule letter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEC = Path(__file__).resolve().parent
CFG_DIR = DEC / "configs" / "rdrift_weight_grid"

# Schedule family (letter) → compression label → (counts, budget W)
SCHEDULES = {
    "A": {
        "778": ([256, 74, 36], 4096),
        "889": ([128, 37, 18], 2048),
        "944": ([64, 17, 10], 1024),
    },
    "B": {
        "778": ([256, 80, 32], 4096),
        "889": ([128, 40, 16], 2048),
        "944": ([64, 20, 8], 1024),
    },
    "C": {
        "778": ([256, 68, 40], 4096),
        "889": ([128, 34, 20], 2048),
        "944": ([64, 17, 10], 1024),
    },
}

# (0.6, 0.4) = production default. Skip re-queue only at 77.8% (hist).
# Still emit W60 at 88.9%/94.4% so each rate has a same-formula baseline.
WEIGHTS_NEW = (0.2, 0.4, 0.8)
WEIGHTS_ALL = (0.2, 0.4, 0.6, 0.8)
WEIGHTS_BY_RATE = {
    "778": WEIGHTS_NEW,  # W60@77.8 reused
    "889": WEIGHTS_ALL,
    "944": WEIGHTS_ALL,
}
RATE_NOTE = {"778": "77.8%", "889": "88.9%", "944": "94.4%"}
LENGTHS = [12, 8, 12]


def arm_id(letter: str, rate: str, w: float) -> str:
    return f"RD_S{letter}_R{rate}_W{int(round(w * 100)):02d}"


def build_cfg(letter: str, rate: str, w: float) -> dict:
    counts, budget = SCHEDULES[letter][rate]
    return {
        "id": arm_id(letter, rate, w),
        "phase": "cvpr2027_rdrift_weight_grid",
        "note": (
            f"RDRIFT weight grid after ABC: schedule {letter}, "
            f"{RATE_NOTE[rate]} W={budget} counts={counts}, "
            f"w={w:g} (1-w={1-w:g}), QA B8, first RDRIFT later Q_ATTN."
        ),
        "method": {
            "original_tokens": 576,
            "depth": 32,
            "budget": budget,
            "schedule": {"counts": counts, "lengths": list(LENGTHS)},
            "stage1": {
                "guide": "dual",
                "cover_family": "legacy_native",
                "cover": True,
                "beta": 0.5,
                "score_guide": "same",
                "payload": "pen",
                "saliency": "BIND_ORIGINAL_SAME_RULE",
            },
            "stage2": {
                "first_selector": "RDRIFT",
                "later_selector": "Q_ATTN",
                "rdrift_current_weight": w,
            },
            "cache_mode": "native",
            "merge": False,
            "learned": False,
            "qa_cover": {
                "entry_mode": "s1_anchor",
                "question_source": "actual",
                "anchor": {
                    "max_swaps": 8,
                    "donor_pool": 16,
                    "drop_pool": 12,
                    "min_raw_std": 0.008,
                    "min_query_gain": 0.08,
                    "min_novelty": 0.05,
                    "max_coverage_loss": 0.005,
                    "coverage_weight": 0.5,
                    "saliency_guard_slots": 0,
                },
                "content_mask": "frozen_template_pieces_v1",
                "seed": 17,
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=("A", "B", "C"), required=True)
    args = ap.parse_args()
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for rate in ("778", "889", "944"):
        for w in WEIGHTS_BY_RATE[rate]:
            cfg = build_cfg(args.schedule, rate, w)
            path = CFG_DIR / f"{cfg['id']}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            written.append(path.name)
        # Always keep W60@778 on disk for hist identity / summary.
        if 0.6 not in WEIGHTS_BY_RATE[rate]:
            cfg60 = build_cfg(args.schedule, rate, 0.6)
            (CFG_DIR / f"{cfg60['id']}.json").write_text(json.dumps(cfg60, indent=2) + "\n")
    manifest = {
        "schedule_letter": args.schedule,
        "weights_by_rate": {k: list(v) for k, v in WEIGHTS_BY_RATE.items()},
        "skip_rerun": {"778": [0.6]},
        "skip_reason": "w=0.6 (0.6/0.4) is production RDRIFT; reuse 77.8% hist only",
        "rates": list(RATE_NOTE.keys()),
        "arms_to_run": written,
        "n_arms_to_run": len(written),
        "schedules": {
            r: {"counts": SCHEDULES[args.schedule][r][0], "W": SCHEDULES[args.schedule][r][1]}
            for r in ("778", "889", "944")
        },
    }
    (CFG_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
