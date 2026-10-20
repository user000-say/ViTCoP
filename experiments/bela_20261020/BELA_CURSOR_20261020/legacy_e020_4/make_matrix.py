"""79 substantive, unique model configurations; use the existing VLM runner."""
import hashlib
import json
from pathlib import Path


def operator(events=(), alpha=.5, variant="default"):
    if not events:
        return {"events": []}
    value = dict(events=list(events), alpha=alpha, assignment="semantic",
                 cosine_floor=.8, radius=2., preserve_norm=False, weights="uniform")
    changes = {
        "default": {}, "semantic_only": dict(radius=None),
        "spatial_only": dict(assignment="spatial", cosine_floor=-1.),
        "random_recipient": dict(assignment="random"),
        "query_weights": dict(weights="existing_question_mass"),
        "norm_restore": dict(preserve_norm=True),
        "loose_gate": dict(cosine_floor=0.),
    }
    if variant not in changes:
        raise ValueError("unknown variant")
    value.update(changes[variant])
    return value


def build():
    rows, keys = [], set()
    def add(block, entry="COVER", profile="SHIFT", first="RDRIFT",
            events=(), alpha=.5, variant="default", final="ORIGINAL", seed=0):
        cfg = dict(entry=entry, profile_alias=profile, first_selector=first,
                   final_selector=final, operator=operator(events, alpha, variant), seed=seed)
        key = json.dumps(cfg, sort_keys=True)
        if key in keys:
            raise AssertionError("duplicate config: " + key)
        keys.add(key)
        rows.append(dict(id=f"{block}_{hashlib.sha256(key.encode()).hexdigest()[:12]}", block=block, **cfg))
    # A: 8 anchors; RDRIFT means the REAL historical FIRST-event rule.
    for entry in ("BASE", "COVER"):
        for profile in ("SHIFT", "NEXT"):
            for first in ("REF", "RDRIFT"):
                add("A", entry=entry, profile=profile, first=first)
    # B: 18 operator interventions at the exact existing schedule and K.
    for first in ("REF", "RDRIFT"):
        for events in (("first",), ("final",), ("first", "final")):
            for alpha in (.25, .5, 1.):
                add("B", first=first, events=events, alpha=alpha)
    # C: 12 controls separating locality, similarity, mass and perturbation.
    for events in (("first",), ("final",)):
        for variant in ("semantic_only", "spatial_only", "random_recipient",
                        "query_weights", "norm_restore", "loose_gate"):
            add("C", events=events, variant=variant)
    # D: 12 cost/frontier tests, each with matched hard-deletion controls.
    for profile in ("SHIFT_K108", "SHIFT_K100", "FINAL11", "FINAL15_WMATCH"):
        for events in ((), ("first",), ("first", "final")):
            add("D", profile=profile, events=events)
    # E: 12 independent final-selector speed controls.
    for profile in ("SHIFT", "SHIFT_K100"):
        for final in ("POOLED_QK", "KEY_ENERGY", "RANDOM"):
            for events in ((), ("first",)):
                add("E", profile=profile, final=final, events=events)
    # F: 6 transfer tests. NEXT is always read from the real old config.
    for entry, profile in (("BASE", "SHIFT"), ("COVER", "NEXT")):
        for events in (("first",), ("final",), ("first", "final")):
            add("F", entry=entry, profile=profile, events=events)
    # G: 6 extra independent random seeds, not six new algorithms.
    for profile in ("SHIFT", "SHIFT_K100"):
        for seed in (1, 2, 3):
            add("G", profile=profile, events=("first",), final="RANDOM", seed=seed)
    # H: independent capacity-response line; overlap with D is reused.
    for entry, first in (("COVER", "REF"), ("COVER", "RDRIFT"), ("BASE", "RDRIFT")):
        for profile in ("SHIFT_K100", "SHIFT_K132"):
            if entry == "COVER" and first == "RDRIFT" and profile == "SHIFT_K100":
                continue
            add("H", entry=entry, first=first, profile=profile)
    assert len(rows) == 79
    return rows


if __name__ == "__main__":
    rows = build()
    output = dict(schema="e020_3_matrix_v1", configs=rows,
        blocks={b: sum(r["block"] == b for r in rows) for b in "ABCDEFGH"},
        profile_semantics="completed block counts; NEXT must bind actual historical JSON",
        selector_semantics="REF/RDRIFT are source-bound first-event paths; ORIGINAL final is unchanged",
        notes=["B/C/D/E can run independently after correctness smoke.",
               "A final random seed is repeated evidence, not a different algorithm.",
               "No learned weights or adaptive-budget premise blocks the queue.",
               "Every extra affinity/QK/gather call must be inside TTFT."])
    Path(__file__).with_name("base_matrix.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"configs":len(rows),"max_four_task_screen_jobs":4*len(rows),"blocks":output["blocks"]}))
