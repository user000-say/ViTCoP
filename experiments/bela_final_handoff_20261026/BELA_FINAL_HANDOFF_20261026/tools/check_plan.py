"""Validate exact decoder budgets and candidate capacity constraints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def budget(schedule, layers=32):
    if not schedule or any(len(x) != 2 for x in schedule):
        raise ValueError("schedule is a sequence of [K, block_count]")
    if any(not isinstance(x, int) or isinstance(x, bool)
           for seg in schedule for x in seg):
        raise ValueError("integer schedule required")
    if any(k < 0 or length <= 0 for k, length in schedule):
        raise ValueError("invalid K or segment length")
    if sum(length for _, length in schedule) != layers:
        raise ValueError("schedule does not cover all decoder blocks")
    if any(schedule[i + 1][0] > schedule[i][0]
           for i in range(len(schedule) - 1)):
        raise ValueError("nested pruning cannot increase K")
    cut_after = []
    elapsed = 0
    for i, (k, length) in enumerate(schedule):
        elapsed += length
        if i + 1 < len(schedule) and schedule[i + 1][0] != k:
            cut_after.append(elapsed)
    return {"W": sum(k * length for k, length in schedule),
            "Q_visual_only": sum(k * k * length for k, length in schedule),
            "cuts_after_1based_blocks": cut_after,
            "entry_K": schedule[0][0], "final_K": schedule[-1][0]}


def permanent_survivors_feasible(schedule, count):
    return 0 <= count <= min(k for k, _ in schedule)


def validate(config):
    results = {k: budget(v, config["layers"])
               for k, v in config["schedules"].items()}
    seen = set()
    for arm in config["arms"]:
        if arm["id"] in seen:
            raise ValueError("duplicate arm ID")
        seen.add(arm["id"])
        info = results[arm["schedule"]]
        if arm["budget"] != info["W"]:
            raise ValueError("arm budget mismatch: " + arm["id"])
        b = arm["admission_cap"]
        if b < 0 or b >= info["entry_K"]:
            raise ValueError("invalid bounded admission cap")
        if arm["cut_exchange"] and arm["anchor_capacity"] <= 0:
            raise ValueError("guard has no finite anchor capacity")
        if arm["role"] != "unequal_cost_diagnostic" and info["W"] not in (1024, 2048):
            raise ValueError("undeclared unequal-cost comparison")
    if results["L2"]["W"] != 1024 or results["E3"]["W"] != 2048:
        raise ValueError("anchor changed")
    return {"status": "PASS", "arms_defined": len(seen),
            "schedule_checks": results,
            "warning": "Q is not full attention FLOPs; no GPU accuracy or speed validated."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config/experiments.json")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = validate(json.loads(args.config.read_text()))
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
