#!/usr/bin/env python3
"""Summarize NeXT-7B A0–A4 v4 ACC: per-task abs/rel, R4 and R6 means."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101/logs/acc_a0_a4_94pct_v5")
DENSE = json.loads(
    Path(
        "/home/xiangyue/ViTCoP/experiments/bela_v100_4model_qacover_20261028/results/next7_dense6_absolute.json"
    ).read_text()
)["absolute"]
DENSE_MAP = {
    "gqa": float(DENSE["gqa"]),
    "pope": float(DENSE["pope_f1"]),
    "mme": float(DENSE["mme"]),
    "mmbench_en_dev": float(DENSE["mmbench_en_dev"]),
    "scienceqa_img": float(DENSE["scienceqa_img"]),
    "textvqa_val_ocr": float(DENSE["textvqa_val_ocr"]),
}
R4 = ("gqa", "pope", "mme", "mmbench_en_dev")
R6 = R4 + ("scienceqa_img", "textvqa_val_ocr")
ARMS = [
    "NEXT7_T160_A0_PAPER",
    "NEXT7_T160_A1_PAPER_VAR",
    "NEXT7_T160_A2_LATE",
    "NEXT7_T160_A3_LESS_S1",
    "NEXT7_T160_A4_STRONG_LATE",
]


def extract_abs(task: str, results: dict) -> float | None:
    block = results.get("results", {}).get(task, {})
    if not isinstance(block, dict):
        return None
    if task == "gqa":
        v = block.get("exact_match,none")
        return None if v is None else 100.0 * float(v)
    if task == "pope":
        v = block.get("pope_f1_score,none")
        return None if v is None else 100.0 * float(v)
    if task == "mme":
        p, c = block.get("mme_perception_score,none"), block.get("mme_cognition_score,none")
        if p is None or c is None:
            return None
        return float(p) + float(c)
    if task == "mmbench_en_dev":
        v = block.get("gpt_eval_score,none")
        return None if v is None else float(v)
    if task in ("scienceqa_img", "textvqa_val_ocr"):
        v = block.get("exact_match,none")
        return None if v is None else 100.0 * float(v)
    return None


def main():
    table = []
    for arm in ARMS:
        row = {"arm": arm, "abs": {}, "rel": {}, "missing": []}
        for t in R6:
            hits = sorted((ROOT / arm / t).glob("**/*_results.json"))
            hits = [p for p in hits if "submission" not in str(p)]
            if not hits:
                row["missing"].append(t)
                continue
            a = extract_abs(t, json.loads(hits[0].read_text()))
            if a is None:
                row["missing"].append(t)
                continue
            row["abs"][t] = a
            row["rel"][t] = 100.0 * a / DENSE_MAP[t]
        if all(t in row["rel"] for t in R4):
            row["R4"] = sum(row["rel"][t] for t in R4) / 4.0
        if all(t in row["rel"] for t in R6):
            row["R6"] = sum(row["rel"][t] for t in R6) / 6.0
        table.append(row)

    out = {
        "dense_ref": DENSE_MAP,
        "note": "rel = 100*abs/NeXT-7B Dense; R4=mean4, R6=mean6 (incl SciQA+TextVQA-OCR)",
        "arms": table,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "SUMMARY_R4_R6.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # compact markdown
    lines = ["# NeXT-7B v4 A0–A4 ACC", "", "| arm | R4 | R6 | missing |", "|---|---:|---:|---|"]
    for r in table:
        lines.append(
            f"| {r['arm']} | {r.get('R4', float('nan')):.2f} | {r.get('R6', float('nan')):.2f} | {','.join(r['missing']) or '-'} |"
        )
    (ROOT / "SUMMARY_R4_R6.md").write_text("\n".join(lines) + "\n")
    print("WROTE", ROOT / "SUMMARY_R4_R6.json")


if __name__ == "__main__":
    main()
