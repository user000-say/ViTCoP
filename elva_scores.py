"""Canonical ELVA / BELA relative-ACC scores (R4 / R8 / R9).

Protocol freeze 2026-10-21:

For every dataset, relative ACC is:

    r = 100 * abs_score / Dense_ref

Dense_ref:
  gqa 61.9, pope 85.8, mme 1852, mmbench 64.1,
  scienceqa_img 69.51, mmmu_val 35.60, ai2d 55.18,
  textvqa_val_ocr 58.20, vizwiz_vqa_val 50.68

Aggregates (arithmetic means of relative ACCs):
  R4 = mean of the 4 R4-task relatives
  R8 = mean of 8 relatives (excludes TextVQA-OCR)
  R9 = mean of all 9 relatives
  R5 = mean of the 5 extra relatives only

Reports must always show both the per-dataset relatives and R9 (and R4/R8).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from elva_acc_tasks import ACC_TASKS, EXTRA_TASKS, R4_TASKS

# Per-task Dense denominators (absolute score units → relative %).
DENSE_REF = {
    "gqa": 61.9,
    "pope": 85.8,
    "mme": 1852.0,
    "mmbench_en_dev": 64.1,
    "scienceqa_img": 69.51,
    "mmmu_val": 35.60,
    "ai2d": 55.18,
    "textvqa_val_ocr": 58.20,
    "vizwiz_vqa_val": 50.68,
}

DENSE_REF_SOURCE = {
    "gqa": "R4_frozen_denom",
    "pope": "R4_frozen_denom_POPE_F1",
    "mme": "R4_frozen_denom",
    "mmbench_en_dev": "R4_frozen_denom",
    "scienceqa_img": "TopV_CVPR2025_Table1_LLaVA15_7B",
    "mmmu_val": "local_same_protocol_dense",
    "ai2d": "TopV_CVPR2025_Table1_LLaVA15_7B",
    "textvqa_val_ocr": "LLaVA_ModelZoo_TextVQA_58.2_OCR_prompt",
    "vizwiz_vqa_val": "local_same_protocol_dense",
}

# Backward-compatible aliases.
R4_DENOM = {t: DENSE_REF[t] for t in R4_TASKS}
DENSE_EXTRA = {t: DENSE_REF[t] for t in EXTRA_TASKS}
DENSE_EXTRA_SOURCE = {t: DENSE_REF_SOURCE[t] for t in EXTRA_TASKS}

R8_TASKS = R4_TASKS + (
    "scienceqa_img",
    "mmmu_val",
    "ai2d",
    "vizwiz_vqa_val",
)
R8_EXTRA_TASKS = (
    "scienceqa_img",
    "mmmu_val",
    "ai2d",
    "vizwiz_vqa_val",
)

TASK_LABEL = {
    "gqa": "GQA",
    "pope": "POPE",
    "mme": "MME",
    "mmbench_en_dev": "MMBench",
    "scienceqa_img": "SciQA-IMG",
    "mmmu_val": "MMMU",
    "ai2d": "AI2D",
    "textvqa_val_ocr": "TextVQA-OCR",
    "vizwiz_vqa_val": "VizWiz",
}


def score_task_from_metrics(m: Mapping[str, Any], task: str) -> float:
    """Extract absolute task score (%) from lmms-eval metrics.json body."""
    if task == "gqa":
        v = float(m["exact_match"])
        return v * (100 if v <= 1.5 else 1)
    if task == "pope":
        v = float(m["pope_f1_score"])
        return v * (100 if v <= 1.5 else 1)
    if task == "mme":
        return float(m["mme_perception_score"]) + float(m["mme_cognition_score"])
    if task == "mmbench_en_dev":
        v = m.get("gpt_eval_score")
        v = float(v if v is not None else m["exact_match"])
        return v * (100 if v <= 1.5 else 1)
    if task == "mmmu_val":
        v = float(m["mmmu_acc"])
        return v * (100 if v <= 1.5 else 1)
    v = float(m["exact_match"])
    return v * (100 if v <= 1.5 else 1)


def relative_retention(abs_score: float, dense: float) -> float:
    return 100.0 * float(abs_score) / float(dense)


def compute_all_relative(tasks: Mapping[str, float]) -> dict[str, float]:
    """Per-dataset relative ACC for every task present that has a Dense ref."""
    out = {}
    for t, abs_score in tasks.items():
        if t not in DENSE_REF:
            continue
        out[t] = relative_retention(abs_score, DENSE_REF[t])
    return out


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals)


def compute_r4(tasks: Mapping[str, float]) -> float:
    """Mean of the 4 R4 relative ACCs (= classic 25*sum formula)."""
    rel = compute_all_relative(tasks)
    return _mean([rel[t] for t in R4_TASKS])


def compute_extra_retention(tasks: Mapping[str, float]) -> dict[str, float]:
    rel = compute_all_relative(tasks)
    return {t: rel[t] for t in EXTRA_TASKS if t in rel}


def compute_r5(retention: Mapping[str, float]) -> float | None:
    vals = [retention[t] for t in EXTRA_TASKS if t in retention]
    if len(vals) != len(EXTRA_TASKS):
        return None
    return _mean(vals)


def compute_r8_from_relative(rel: Mapping[str, float]) -> float | None:
    if any(t not in rel for t in R8_TASKS):
        return None
    return _mean([rel[t] for t in R8_TASKS])


def compute_r9_from_relative(rel: Mapping[str, float]) -> float | None:
    if any(t not in rel for t in ACC_TASKS):
        return None
    return _mean([rel[t] for t in ACC_TASKS])


def compute_r8(r4: float, retention: Mapping[str, float]) -> float | None:
    # Kept for callers; equivalent to mean of 8 relatives when r4 is R4 mean.
    if any(t not in retention for t in R8_EXTRA_TASKS):
        return None
    return (4.0 * float(r4) + sum(retention[t] for t in R8_EXTRA_TASKS)) / 8.0


def compute_r9(r4: float, retention: Mapping[str, float]) -> float | None:
    if any(t not in retention for t in EXTRA_TASKS):
        return None
    return (4.0 * float(r4) + sum(retention[t] for t in EXTRA_TASKS)) / 9.0


def summarize_arm(tasks: Mapping[str, float]) -> dict[str, Any]:
    """Absolute scores + per-dataset relative ACC + R4/R8/R9 means."""
    rel = compute_all_relative(tasks)
    row: dict[str, Any] = {
        "tasks_abs": {t: float(tasks[t]) for t in tasks},
        "dense_ref": dict(DENSE_REF),
        "dense_ref_source": dict(DENSE_REF_SOURCE),
        "relative_acc": {t: rel[t] for t in ACC_TASKS if t in rel},
        "retention_vs_dense": {t: rel[t] for t in EXTRA_TASKS if t in rel},
        "dense_extra": dict(DENSE_EXTRA),
        "dense_extra_source": dict(DENSE_EXTRA_SOURCE),
    }
    if all(t in rel for t in R4_TASKS):
        row["R4"] = _mean([rel[t] for t in R4_TASKS])
    if all(t in rel for t in EXTRA_TASKS):
        row["R5"] = _mean([rel[t] for t in EXTRA_TASKS])
    if all(t in rel for t in R8_TASKS):
        row["R8"] = _mean([rel[t] for t in R8_TASKS])
    if all(t in rel for t in ACC_TASKS):
        row["R9"] = _mean([rel[t] for t in ACC_TASKS])
    row["formulas"] = {
        "r_task": "100 * abs / Dense_ref  (e.g. SciQA/69.51, AI2D/55.18)",
        "R4": "mean(r_GQA, r_POPE, r_MME, r_MMBench)",
        "R8": "mean of 8 relatives (excludes TextVQA-OCR)",
        "R9": "mean of all 9 relatives",
        "R5": "mean of 5 extra relatives",
        "note": (
            "Tables must list every dataset relative_acc plus R9. "
            "Dense refs mixed local+literature until unified Dense table."
        ),
    }
    return row


def _fmt_abs_rel(abs_score: float | None, rel: float | None, task: str) -> str:
    """e.g. GQA 57.84/93.44%  or MME 1765.2/95.31%."""
    if abs_score is None or rel is None:
        return "—"
    if task == "mme":
        return f"{abs_score:.1f}/{rel:.2f}%"
    return f"{abs_score:.2f}/{rel:.2f}%"


def format_relative_table(
    arms: Mapping[str, Mapping[str, Any]],
    arm_order: Sequence[str] | None = None,
) -> str:
    """Markdown table: each cell is abs/rel% ; footer rows are means."""
    order = list(arm_order) if arm_order else list(arms.keys())
    headers = ["dataset", "Dense"] + list(order)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for t in ACC_TASKS:
        cells = [TASK_LABEL.get(t, t), f"{DENSE_REF[t]:g}"]
        for arm in order:
            block = arms[arm]
            abs_map = block.get("tasks_abs") or block.get("tasks") or {}
            rel_map = block.get("relative_acc") or {}
            cells.append(_fmt_abs_rel(abs_map.get(t), rel_map.get(t), t))
        lines.append("| " + " | ".join(cells) + " |")
    for name, key in (
        ("R4 mean (rel%)", "R4"),
        ("R8 mean (rel%)", "R8"),
        ("R9 mean (rel%)", "R9"),
        ("R5 extras (rel%)", "R5"),
    ):
        cells = [name, "—"]
        for arm in order:
            v = arms[arm].get(key)
            cells.append(f"{v:.2f}%" if isinstance(v, (int, float)) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
