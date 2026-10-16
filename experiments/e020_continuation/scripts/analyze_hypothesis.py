#!/usr/bin/env python3
"""Summarize E020 + E019 BC results and emit a hypothesis verdict."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))
import e014_arms as m  # noqa: E402

DENOM = {"gqa": 61.9, "pope": 85.8, "mme": 1852.0, "mmbench_en_dev": 64.1}
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
ANCHOR = 96.50


def pull(root: Path, arm: str):
    vals = {}
    for t in TASKS:
        p = root / arm / t / "metrics.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        if t == "gqa":
            vals[t] = 100 * float(d["exact_match"])
        elif t == "pope":
            vals[t] = 100 * float(d["pope_accuracy"])
        else:
            r = json.loads(Path(d["results_json"]).read_text())["results"][t]
            if t == "mme":
                vals[t] = (
                    r["mme_perception_score,none"] + r["mme_cognition_score,none"]
                )
            else:
                vals[t] = r["gpt_eval_score,none"]
    r4 = 25 * sum(vals[t] / DENOM[t] for t in TASKS)
    return {"tasks": vals, "R4": r4, "delta": r4 - ANCHOR}


def table(title: str, rows: list[tuple[str, dict]]):
    lines = [f"## {title}", ""]
    lines.append(f"| arm | GQA | POPE | MMB | MME | R4 | vs {ANCHOR:.2f} |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm, s in sorted(rows, key=lambda x: -x[1]["R4"]):
        v = s["tasks"]
        lines.append(
            f"| {arm} | {v['gqa']:.2f} | {v['pope']:.2f} | "
            f"{v['mmbench_en_dev']:.2f} | {v['mme']:.1f} | {s['R4']:.2f} | "
            f"{s['delta']:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    e020_root = REPO / "logs/e014_aee/full/full_e020_priority"
    e020_learned_root = REPO / "logs/e014_aee/full/full_e020_learned"
    e019_root = REPO / "logs/e014_aee/full/full_e019_remaining"
    models = REPO / "experiments/e020_continuation/models"

    e020_done, e020_pend = [], []
    for arm in m.FULL_E020_PRIORITY:
        s = pull(e020_root, arm)
        (e020_done if s else e020_pend).append((arm, s) if s else arm)

    learned_done, learned_pend = [], []
    for arm in m.FULL_E020_LEARNED:
        s = pull(e020_learned_root, arm)
        (learned_done if s else learned_pend).append((arm, s) if s else arm)

    e019_done, e019_pend = [], []
    for arm in m.FULL_E019_REMAINING:
        s = pull(e019_root, arm)
        (e019_done if s else e019_pend).append((arm, s) if s else arm)

    ref = [x for x in e020_done if x[0].endswith("_REF32")]
    ctrl = [x for x in e020_done if not x[0].endswith("_REF32")]
    cover_shift = next((s for a, s in ref if a == "E020_COVER_FIRST_SHIFT_REF32"), None)
    best_ctrl = max(ctrl, key=lambda x: x[1]["R4"]) if ctrl else None
    best_learned = max(learned_done, key=lambda x: x[1]["R4"]) if learned_done else None

    # E020 verdict
    e020_status = "INCOMPLETE"
    e020_verdict = []
    if cover_shift is None:
        e020_verdict.append("Missing COVER FIRST_SHIFT REF anchor.")
    else:
        e020_verdict.append(
            f"Anchor COVER_FIRST_SHIFT_REF R4={cover_shift['R4']:.2f} "
            f"(target {ANCHOR:.2f})."
        )
    if not ctrl:
        e020_verdict.append(
            "No D-control full rows yet → cannot test whether simple first-event "
            "exchanges beat the fixed schedule."
        )
    else:
        beat = [a for a, s in ctrl if s["R4"] > ANCHOR + 0.05]
        if beat:
            e020_verdict.append(
                f"D-controls beating anchor: {', '.join(beat)}. "
                "Simple first-event rules may help; learned module must still clear them."
            )
        else:
            assert best_ctrl is not None
            e020_verdict.append(
                f"No D-control clears {ANCHOR:.2f} (best {best_ctrl[0]} "
                f"R4={best_ctrl[1]['R4']:.2f}). Simple exchanges do not prove the claim."
            )
    model_files = list(models.glob("*.npz")) if models.is_dir() else []
    if not learned_done:
        if not model_files:
            e020_verdict.append(
                "Learned B/C not runnable: no fitted PairScorer checkpoints "
                "(pairdata/fit pending). Core E020 conjecture NOT tested."
            )
            e020_status = "PARTIAL_NOLEARN_ONLY"
        else:
            e020_verdict.append(
                "Models exist but learned ACC incomplete. Core conjecture still open."
            )
            e020_status = "PARTIAL_LEARNED_PENDING"
    else:
        assert best_learned is not None
        if best_learned[1]["R4"] > ANCHOR + 0.05 and (
            not best_ctrl or best_learned[1]["R4"] > best_ctrl[1]["R4"] + 0.05
        ):
            e020_verdict.append(
                f"Learned best {best_learned[0]} R4={best_learned[1]['R4']:.2f} "
                f"beats anchor and best control → SUPPORTS E020 conjecture "
                "(still need paired CIs / TTFT)."
            )
            e020_status = "SUPPORTIVE"
        else:
            e020_verdict.append(
                f"Learned best {best_learned[0]} R4={best_learned[1]['R4']:.2f} "
                "does not clearly beat anchor+controls → does NOT support the conjecture."
            )
            e020_status = "NOT_SUPPORTED"

    # E019 BC: beat matching schedule REF?
    e019_verdict = []
    e019_status = "INCOMPLETE"
    if not e019_done:
        e019_verdict.append("E019 B/C remaining ACC not finished.")
    else:
        # Compare selector arms to Block-A schedule refs when available.
        block_a = REPO / "experiments/e019_progressive_evidence/BLOCK_A_RESULTS.json"
        refs = json.loads(block_a.read_text())["acc"] if block_a.exists() else {}
        wins = 0
        compared = 0
        for arm, s in e019_done:
            # Map to schedule ref key if possible
            key = None
            for sched in ("FIRST_SHIFT_WMATCH", "THREE_WMATCH", "LATE_WMATCH", "EARLY"):
                if f"_{sched}_" in arm or arm.endswith(f"_{sched}32"):
                    stage = "COVER" if "_COVER_" in arm else "BASE"
                    # Block A uses E019_COVER_FIRST_SHIFT_WMATCH32 style for WMATCH
                    if sched == "EARLY":
                        key = f"{stage}_EARLY"
                    else:
                        key = f"E019_{stage}_{sched}32"
                    break
            if key and key in refs:
                compared += 1
                if s["R4"] > refs[key]["R4"] + 0.05:
                    wins += 1
        e019_verdict.append(
            f"Finished {len(e019_done)}/{len(m.FULL_E019_REMAINING)} remaining arms; "
            f"{wins}/{compared} beat matched Block-A schedule REF by >0.05 R4."
        )
        if compared == 0:
            e019_status = "PARTIAL"
        elif wins == 0:
            e019_status = "NOT_SUPPORTED"
            e019_verdict.append(
                "Progressive-evidence selectors do not beat strong fixed schedules → "
                "E019 B/C conjecture not supported on current ACC."
            )
        elif wins >= max(1, compared // 3):
            e019_status = "SUPPORTIVE"
            e019_verdict.append(
                "Multiple selector arms beat schedule REF → supportive but needs "
                "paired stats / which tasks drive gains."
            )
        else:
            e019_status = "MIXED"

    out = {
        "schema": "hypothesis_verdict_v1",
        "anchor_R4": ANCHOR,
        "e020": {
            "status": e020_status,
            "priority_done": len(e020_done),
            "priority_total": len(m.FULL_E020_PRIORITY),
            "learned_done": len(learned_done),
            "learned_total": len(m.FULL_E020_LEARNED),
            "models_present": [p.name for p in model_files],
            "verdict": e020_verdict,
            "best_ref": max(ref, key=lambda x: x[1]["R4"])[0] if ref else None,
            "best_control": best_ctrl[0] if best_ctrl else None,
            "best_learned": best_learned[0] if best_learned else None,
        },
        "e019": {
            "status": e019_status,
            "remaining_done": len(e019_done),
            "remaining_total": len(m.FULL_E019_REMAINING),
            "verdict": e019_verdict,
        },
        "overall": (
            "PROVEN"
            if e020_status == "SUPPORTIVE" and e019_status in {"SUPPORTIVE", "MIXED", "NOT_SUPPORTED"}
            else (
                "NOT_PROVEN"
                if "NOT_SUPPORTED" in (e020_status, e019_status)
                or e020_status.startswith("PARTIAL")
                or e019_status == "INCOMPLETE"
                else "OPEN"
            )
        ),
    }

    md = ["# Hypothesis verdict", ""]
    md.append(f"**Overall: {out['overall']}**")
    md.append("")
    md.append("### E020")
    md.append(f"- status: `{e020_status}`")
    for line in e020_verdict:
        md.append(f"- {line}")
    md.append("")
    md.append("### E019 B/C")
    md.append(f"- status: `{e019_status}`")
    for line in e019_verdict:
        md.append(f"- {line}")
    md.append("")
    if ref:
        md.append(table("E020 REF (completed)", ref))
    if ctrl:
        md.append(table("E020 D controls (completed)", ctrl))
    if learned_done:
        md.append(table("E020 learned B/C (completed)", learned_done))
    if e019_done:
        md.append(table("E019 remaining B/C (completed)", e019_done))

    pkg = REPO / "experiments/e020_continuation"
    (pkg / "HYPOTHESIS_VERDICT.json").write_text(json.dumps(out, indent=2))
    (pkg / "HYPOTHESIS_VERDICT.md").write_text("\n".join(md))
    print(json.dumps(out, indent=2))
    print("---")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
