"""Generate finite, deduplicated development configs. No VLM is run here."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from elva.budget import Schedule


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_catalog():
    result, seen = [], set()

    def add(block, priority, budget, schedule, guide="pen", cover=True,
            selector="Q_ATTN", beta=.5, final_selector="Q_ATTN", seed=0, conditional=False,
            score_guide="same"):
        assert schedule.work == budget and schedule.depth == 32
        method = dict(budget=budget, original_tokens=576, depth=32,
                      schedule=dict(counts=schedule.counts, lengths=schedule.lengths),
                      stage1=dict(guide=guide, cover=cover, beta=beta if guide == "dual" or score_guide == "dual" else None,
                                  score_guide=score_guide, payload="pen", saliency="BIND_ORIGINAL_SAME_RULE"),
                      stage2=dict(first_selector=selector, later_selector=final_selector,
                                  needs_l9_features=selector == "TRANSIENT_PAIR_PLUS_L9",
                                  seed=seed if "RANDOM" in final_selector else None),
                      merge=False, cache_mode="native", conditional=conditional)
        digest = fingerprint(method)
        if digest in seen:
            return
        seen.add(digest)
        result.append(dict(id=f"ELVA_R2_{len(result):03d}", block=block, priority=priority,
                           config_hash=digest, method=method))

    for budget, counts in [(1024, (60, 28, 12)), (1536, (90, 42, 18)), (2048, (120, 56, 24))]:
        k0, middle, final = counts
        long = Schedule(counts, (10, 10, 12))
        for guide, cover in [("pen", False), ("l9", False), ("dual", False), ("pen", True), ("dual", True)]:
            add("A_stage1", 0 if budget == 1536 else 1, budget, long, guide, cover)
        shapes = [long,
                  Schedule((k0, middle, final), (8, 16, 8)),
                  Schedule((k0, 32*budget//1024, final), (10, 8, 14)),
                  Schedule((k0, 20*budget//1024, final), (12, 8, 12))]
        for shape in shapes:
            for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
                add("B_schedule", 1, budget, shape, selector=selector)
        add("J_stage1_stage2_bridge", 2, budget, long, guide="dual", selector="RDRIFT_NATIVE")
        legacy = {1024: (80, 48, 16), 1536: (96, 64, 32), 2048: (160, 96, 32)}[budget]
        for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
            add("C_legacy_recipe", 1, budget, Schedule(legacy, (3, 10, 19)), selector=selector)
            add("H_stage1_transfer", 2, budget, Schedule(legacy, (3, 10, 19)),
                guide="dual", selector=selector)
        for selector in ("QED_NATIVE", "ATTN_COVER_NATIVE"):
            add("D_selector", 1, budget, long, selector=selector)
        for seed in (11, 23, 47):
            add("E_random_final", 2, budget, long, final_selector="RANDOM_FINAL", seed=seed)
        for beta in (.25, .75):
            add("F_beta", 2, budget, long, guide="dual", beta=beta)
        if budget == 1536:
            add("I_stage1_factor", 2, budget, long, guide="dual", score_guide="pen")
            add("I_stage1_factor", 2, budget, long, guide="pen", score_guide="dual")
        # Conditional heads use forced-final identity F, including their own
        # matched anchor. They CANNOT be compared only to natural re-selection.
        for selector in ("FROZEN_FINAL_ANCHOR", "TRANSIENT_PAIR_CURRENT", "TRANSIENT_PAIR_PLUS_L9"):
            add("G_conditional", 3, budget, long, selector=selector, conditional=True)
    return sorted(result, key=lambda c: (c["priority"], c["id"]))


def main():
    root = Path(__file__).resolve().parent
    out = root/"configs"
    out.mkdir(exist_ok=True)
    catalog = build_catalog()
    for config in catalog:
        (out/f"{config['id']}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n")
    summary = dict(configs=len(catalog), blocks=dict(Counter(c["block"] for c in catalog)),
                   unconditional=sum(not c["method"]["conditional"] for c in catalog),
                   conditional=sum(c["method"]["conditional"] for c in catalog),
                   four_task_jobs=4*len(catalog),
                   note="development capacity, not completed experiments; native source aliases must be deduplicated")
    (out/"catalog.json").write_text(json.dumps(dict(summary=summary, configs=catalog), indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

