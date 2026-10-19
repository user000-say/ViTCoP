"""Two budgets only. Stage1 first; Stage2 requires an explicit result freeze."""
import argparse
from pathlib import Path
import json
import hashlib
import copy
from elva.budget import Schedule


TASKS = ("GQA", "POPE", "MME", "MMBench")
MAIN = {2048: Schedule((172, 90, 32), (4, 8, 20)),
        1024: Schedule((60, 28, 12), (10, 10, 12))}
SHAPES = {2048: (MAIN[2048], Schedule((172, 61, 32), (4, 16, 12)),
                 Schedule((172, 94, 32), (2, 12, 18))),
          1024: (MAIN[1024], Schedule((60, 28, 12), (8, 16, 8)),
                 Schedule((60, 32, 12), (10, 8, 14)))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_config(budget, stage, schedule, *, phase, selector="Q_ATTN", label=""):
    if budget not in MAIN or schedule.work != budget or schedule.depth != 32:
        raise ValueError("only exact W2048 / W1024 576x32 schedules are permitted")
    method = dict(original_tokens=576, depth=32, budget=budget,
                  schedule=dict(counts=list(schedule.counts), lengths=list(schedule.lengths)),
                  stage1=copy.deepcopy(stage), stage2=dict(first_selector=selector, later_selector="Q_ATTN"),
                  cache_mode="native", merge=False, learned=False)
    sha = digest(method)
    return dict(id=f"R3_W{budget}_{label}_{sha[:10]}", phase=phase, config_hash=sha, method=method)


def stage(guide, cover_family="none", beta=None, score_guide="same", slots=12):
    value = dict(guide=guide, cover_family=cover_family, cover=cover_family != "none",
                beta=beta if guide == "dual" or score_guide == "dual" else None,
                score_guide=score_guide, payload="pen", saliency="BIND_ORIGINAL_SAME_RULE")
    if cover_family == "protected_coverage":
        from elva.low_budget_cover import POLICY
        value.update(coverage_slots=slots, coverage_policy=POLICY)
    return value


def build_catalog(*, low_cover="protected_coverage", decisions=None, coverage_base=None):
    if low_cover not in ("protected_coverage", "legacy_pool172_then_topk"):
        raise ValueError("explicit K60 COVER strategy required")
    result, seen = [], set()
    def add(config):
        key = (config["phase"], config["config_hash"])
        if key not in seen:
            result.append(config)
            seen.add(key)
    for budget in (2048, 1024):
        family = "legacy_native" if budget == 2048 else "protected_coverage"
        for name, guide, cover, beta in (("V1", "pen", "none", None), ("V2", "l9", "none", None),
                                         ("V3", "dual", "none", .5), ("V4", "pen", family, None),
                                         ("V5", "dual", family, .5)):
            suffix = ("_POOL172" if cover == "legacy_pool172_then_topk" else
                      "_PCOV60_B12" if cover == "protected_coverage" else "")
            add(make_config(budget, stage(guide, cover, beta), MAIN[budget],
                            phase="stage1_core", label=name+suffix))
        for name, guide, beta in (("B025", "dual", .25), ("B075", "dual", .75), ("L9_COVER", "l9", None)):
            add(make_config(budget, stage(guide, family, beta), MAIN[budget],
                            phase="stage1_optional", label=name))
        for name, guide, score in (("GEO_ONLY", "dual", "pen"), ("SCORE_ONLY", "pen", "dual")):
            add(make_config(budget, stage(guide, family, .5, score_guide=score), MAIN[budget],
                            phase="stage1_optional", label=name))
    # Only execute these on the winning guide, if core evidence warrants it.
    # The unchanged PORT60 implementation is a separate source-bound control.
    bases = [stage("pen","protected_coverage"),stage("dual","protected_coverage",.5)]
    if coverage_base is not None:
        if (digest(coverage_base.get("method",{})) != coverage_base.get("config_hash")
                or coverage_base['method'].get('budget') != 1024
                or coverage_base['method']['stage1'].get('cover_family') != 'protected_coverage'
                or coverage_base['method']['stage1'].get('coverage_slots') != 12):
            raise ValueError('coverage refinement requires the actual fingerprinted W1024 PCOV/b12 base config')
        bases = [copy.deepcopy(coverage_base['method']['stage1'])]
    for base in bases:
        guide = base['guide']
        for slots in (6, 18):
            refined=copy.deepcopy(base);refined['coverage_slots']=slots
            add(make_config(1024, refined, MAIN[1024],
                            phase="stage1_optional", label=f"PCOV_{guide}_B{slots}"))
        port=copy.deepcopy(base);port['cover_family']='parameterized_native'
        port.pop('coverage_policy',None);port.pop('coverage_slots',None)
        add(make_config(1024, port, MAIN[1024],
                        phase="stage1_optional", label=f"COVER_PORT60_{guide}_B11"))
    if low_cover == "legacy_pool172_then_topk":
        # Explicit pilot-only diagnostic. It can be identically top-k when
        # legacy COVER changes only the tail of its Top172 pool.
        for name, guide, beta in (("V4", "pen", None), ("V5", "dual", .5)):
            add(make_config(1024, stage(guide, "legacy_pool172_then_topk", beta), MAIN[1024],
                            phase="diagnostic_pool", label="POOL172_CONTROL_"+name))
    if decisions:
        for budget in (2048, 1024):
            item = decisions.get("budgets", {}).get(str(budget))
            if not item:
                continue
            # Preserve a completed refined winner when regenerating Stage2.
            # The workflow revalidates its actual four-task evidence hashes.
            if item.get('winner_config'):
                winner=item['winner_config']
                if digest(winner.get('method',{})) != winner.get('config_hash'):
                    raise ValueError('edited Stage1 winner config')
                add(copy.deepcopy(winner))
            frozen = item["stage1"]
            for shape_index, schedule in enumerate(SHAPES[budget]):
                for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
                    add(make_config(budget, frozen, schedule, phase="stage2",
                                    selector=selector, label=f"S{shape_index}_{selector}"))
            for selector in ("QED_NATIVE", "ATTN_COVER_NATIVE"):
                add(make_config(budget, frozen, MAIN[budget], phase="stage2",
                                selector=selector, label=selector))
    return result


def write_catalog(output, *, low_cover="protected_coverage", decisions=None, coverage_base=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = build_catalog(low_cover=low_cover, decisions=decisions, coverage_base=coverage_base)
    prior_path=output/"catalog.json"
    if prior_path.exists():
        previous=json.loads(prior_path.read_text())
        keep={c["id"] for c in configs}
        for old in previous.get("configs", []):
            if old.get("id", "").startswith("R3_") and old["id"] not in keep:
                stale=output/(old["id"]+".json")
                if stale.is_file():
                    if json.loads(stale.read_text()).get("config_hash") != old.get("config_hash"):
                        raise ValueError("edited stale generated config; move it before regenerating")
                    stale.unlink()
    for config in configs:
        (output/(config["id"]+".json")).write_text(json.dumps(config, indent=2)+"\n")
    catalog = dict(schema="elva_r3_two_budget", configs=configs,
                   budgets=[2048, 1024], default_phase="stage1_core", low_cover="protected_coverage",
                   include_pool_diagnostic=low_cover=="legacy_pool172_then_topk")
    (output/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="configs")
    parser.add_argument("--include-pool-diagnostic", action="store_true")
    parser.add_argument("--decisions")
    parser.add_argument("--coverage-refine-from", help="actual W1024 PCOV/b12 config after guide/beta choice")
    args = parser.parse_args()
    decisions = json.loads(Path(args.decisions).read_text()) if args.decisions else None
    coverage_base = json.loads(Path(args.coverage_refine_from).read_text()) if args.coverage_refine_from else None
    value = write_catalog(args.out, low_cover="legacy_pool172_then_topk" if args.include_pool_diagnostic else "protected_coverage", decisions=decisions,coverage_base=coverage_base)
    print(json.dumps({"configs": len(value["configs"]), "core": sum(c["phase"]=="stage1_core" for c in value["configs"]), "budgets":value["budgets"]}))
