"""BELA cross-backbone budget planner. Standard-library only.

The schedules named "STAR-Pro" are transcribed from the paper's Table 5.
A schedule scales by *observed post-merge visual N*, not raw ViT patch count.
Layer-average budget W is a comparison proxy, NOT real FLOPs or TTFT.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path


SPEC_FILE = Path(__file__).resolve().parents[1] / "configs/model_budget_specs.json"


def load_specs(path=SPEC_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Budget:
    model: str
    tier: str
    kind: str
    N: int
    L: int
    cuts: tuple[int, ...]
    counts: tuple[int, ...]
    W: int
    dense_W: int
    keep_fraction: float
    reduction_fraction: float
    avg_visual_tokens: float
    target_avg: float

    def json_dict(self):
        return self.__dict__


def round_positive_scaled(x: int, observed_n: int, ref_n: int):
    # Round-to-nearest with explicit half-up behavior, independent of Python's bankers rounding.
    return max(1, (2*x*observed_n + ref_n) // (2*ref_n))


def compute_budget(model: str, tier: str, observed_n: int, specs=None) -> Budget:
    specs = specs or load_specs()
    if model not in specs:
        raise ValueError(f"Unknown model: {model}")
    spec = specs[model]
    if tier not in spec["budgets"]:
        raise ValueError(f"No tier {tier} for {model}")
    if observed_n <= 0:
        raise ValueError("observed_n must be > 0")
    ref_n = spec["reference_n"]
    layers = spec["decoder_layers"]
    setting = spec["budgets"][tier]
    cut = tuple(setting["cuts"])
    stages = tuple([0] + list(cut) + [layers])
    if not (all(0 < c < layers for c in cut) and list(cut) == sorted(set(cut))):
        raise ValueError("invalid cuts")
    raw_counts = [round_positive_scaled(k, observed_n, ref_n) for k in setting["counts"]]
    # All counts are cardinalities of ORIGINAL post-merger visual IDs (not separator tokens).
    counts = []
    previous = observed_n
    for count in raw_counts:
        count = min(count, previous)
        counts.append(count)
        previous = count
    if len(counts) != len(stages)-1:
        raise ValueError("count/cut count mismatch")
    W = sum((stages[i+1] - stages[i]) * counts[i] for i in range(len(counts)))
    dense = observed_n * layers
    ideal_t = setting["target_t"] * observed_n / ref_n if setting["kind"] == "layer_average" \
        else setting["target_k"] * observed_n / ref_n
    return Budget(model, tier, setting["kind"], observed_n, layers, cut,
                  tuple(counts), W, dense, W/dense, 1-W/dense, W/layers, ideal_t)


def allocate_group_quotas(total: int, group_sizes: list[int], minimum_per_group: int = 0) -> list[int]:
    """Exact sum via largest remainder; cannot exceed each group's size."""
    if total < 0 or any(size < 0 for size in group_sizes):
        raise ValueError("negative values")
    if not group_sizes:
        if total == 0:
            return []
        raise ValueError("nonzero total for empty groups")
    if total > sum(group_sizes):
        raise ValueError("more tokens than candidate capacity")
    nonempty = [i for i,s in enumerate(group_sizes) if s > 0]
    if minimum_per_group and len(nonempty)*minimum_per_group > total:
        raise ValueError("not enough budget to retain per-group minimum")
    for i in nonempty:
        if group_sizes[i] < minimum_per_group:
            raise ValueError("group smaller than required minimum")
    base = [min(minimum_per_group, s) for s in group_sizes]
    remaining = total-sum(base)
    caps = [s-b for s,b in zip(group_sizes,base)]
    while remaining:
        available = [i for i,c in enumerate(caps) if c]
        if not available:
            raise ValueError("cannot allocate remaining budget")
        cap_sum = sum(caps[i] for i in available)
        numerators = [(i,remaining*caps[i]/cap_sum) for i in available]
        grants = [0]*len(caps)
        for i,share in numerators:
            grants[i] = min(caps[i], int(share))
        used = sum(grants)
        rest = remaining-used
        ranks = sorted(available, key=lambda i: (-(remaining*caps[i]/cap_sum-int(remaining*caps[i]/cap_sum)), i))
        for i in ranks:
            if not rest:
                break
            if grants[i] < caps[i]:
                grants[i] += 1
                rest -= 1
        assert sum(grants) > 0, "allocation must make progress"
        for i,add in enumerate(grants):
            base[i] += add
            caps[i] -= add
        remaining -= sum(grants)
    assert sum(base) == total
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(load_specs()), required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--observed-n", type=int, required=True)
    ap.add_argument("--groups", type=str, help="comma separated actual post-merge crop lengths")
    args = ap.parse_args()
    b=compute_budget(args.model,args.tier,args.observed_n)
    out=b.json_dict()
    if args.groups:
        groups=[int(i) for i in args.groups.split(",")]
        if sum(groups) != args.observed_n:
            raise ValueError("groups must sum to observed_n, excluding separators")
        out["entry_group_quotas"] = allocate_group_quotas(b.counts[0],groups,minimum_per_group=1)
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
