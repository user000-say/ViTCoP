"""Canonical ELVA / BELA retention scores (R4 / R8 / R9).

Frozen from 2026-10-21 user protocol:

- R4: original four-task macro (unchanged denominators).
- Extra absolute accuracies stay reported as-is.
- Relative retention r = 100 * abs / Dense_ref for the five extras.
- Dense refs mix same-protocol local Dense (MMMU, VizWiz) and literature
  Dense for ScienceQA / AI2D / TextVQA-OCR until a full unified Dense table exists.
- R8 = (4*R4 + r_SciQA + r_MMMU + r_AI2D + r_VizWiz) / 8
- R9 = (4*R4 + r_SciQA + r_MMMU + r_AI2D + r_TextVQA + r_VizWiz) / 9
- R5 = mean of the five extra relative retentions (diagnostic only).

Do not fold absolute extras into R4. Do not silently change Dense refs.
"""
from __future__ import annotations

from typing import Any, Mapping

from elva_acc_tasks import EXTRA_TASKS, R4_TASKS

# R4 Dense / paper denominators (frozen).
R4_DENOM = {
    "gqa": 61.9,
    "pope": 85.8,
    "mme": 1852.0,
    "mmbench_en_dev": 64.1,
}

# Provisional Dense baselines for extras (user freeze 2026-10-21).
# Sources recorded for audit; not a claim of unified protocol.
DENSE_EXTRA = {
    "scienceqa_img": 69.51,       # CVPR 2025 TopV LLaVA-1.5-7B Dense
    "mmmu_val": 35.60,            # local same-protocol Dense
    "ai2d": 55.18,                # CVPR 2025 TopV LLaVA-1.5-7B Dense
    "textvqa_val_ocr": 58.20,     # LLaVA Model Zoo / LLaVA-MORE OCR prompt
    "vizwiz_vqa_val": 50.68,      # local same-protocol Dense
}

DENSE_EXTRA_SOURCE = {
    "scienceqa_img": "TopV_CVPR2025_Table1_LLaVA15_7B",
    "mmmu_val": "local_same_protocol_dense",
    "ai2d": "TopV_CVPR2025_Table1_LLaVA15_7B",
    "textvqa_val_ocr": "LLaVA_ModelZoo_TextVQA_58.2_OCR_prompt",
    "vizwiz_vqa_val": "local_same_protocol_dense",
}

# Tasks that enter exploratory R8 (TextVQA-OCR excluded from R8, kept in R9).
R8_EXTRA_TASKS = (
    "scienceqa_img",
    "mmmu_val",
    "ai2d",
    "vizwiz_vqa_val",
)


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


def compute_r4(tasks: Mapping[str, float]) -> float:
    return 25.0 * sum(float(tasks[t]) / R4_DENOM[t] for t in R4_TASKS)


def relative_retention(abs_score: float, dense: float) -> float:
    return 100.0 * float(abs_score) / float(dense)


def compute_extra_retention(tasks: Mapping[str, float]) -> dict[str, float]:
    out = {}
    for t in EXTRA_TASKS:
        if t not in tasks:
            continue
        out[t] = relative_retention(tasks[t], DENSE_EXTRA[t])
    return out


def compute_r5(retention: Mapping[str, float]) -> float | None:
    vals = [retention[t] for t in EXTRA_TASKS if t in retention]
    if len(vals) != len(EXTRA_TASKS):
        return None
    return sum(vals) / 5.0


def compute_r8(r4: float, retention: Mapping[str, float]) -> float | None:
    if any(t not in retention for t in R8_EXTRA_TASKS):
        return None
    return (4.0 * float(r4) + sum(retention[t] for t in R8_EXTRA_TASKS)) / 8.0


def compute_r9(r4: float, retention: Mapping[str, float]) -> float | None:
    if any(t not in retention for t in EXTRA_TASKS):
        return None
    return (4.0 * float(r4) + sum(retention[t] for t in EXTRA_TASKS)) / 9.0


def summarize_arm(tasks: Mapping[str, float]) -> dict[str, Any]:
    """Build absolute + retention + R4/R8/R9 block from absolute task scores."""
    row: dict[str, Any] = {
        "tasks_abs": {t: float(tasks[t]) for t in tasks},
        "dense_extra": dict(DENSE_EXTRA),
        "dense_extra_source": dict(DENSE_EXTRA_SOURCE),
    }
    if all(t in tasks for t in R4_TASKS):
        row["R4"] = compute_r4(tasks)
    retention = compute_extra_retention(tasks)
    row["retention_vs_dense"] = retention
    if "R4" in row:
        row["R5"] = compute_r5(retention)
        row["R8"] = compute_r8(row["R4"], retention)
        row["R9"] = compute_r9(row["R4"], retention)
        row["formulas"] = {
            "R4": "25*(GQA/61.9 + POPE_F1/85.8 + MME/1852 + MMBench/64.1)",
            "R8": "(4*R4 + r_SciQA + r_MMMU + r_AI2D + r_VizWiz)/8",
            "R9": "(4*R4 + r_SciQA + r_MMMU + r_AI2D + r_TextVQA + r_VizWiz)/9",
            "R5": "mean(r_SciQA, r_MMMU, r_AI2D, r_TextVQA, r_VizWiz)",
            "r_task": "100 * abs / Dense_ref",
            "note": (
                "R8 exploratory (excludes TextVQA-OCR). Dense refs mixed "
                "local+literature until unified same-protocol Dense table."
            ),
        }
    return row
