"""Canonical ACC task sets for ELVA / CVPR runs.

R4 stays the original four-task macro. The four extras are reported separately
and must be queued on every full ACC campaign from 2026-10-20 onward.
"""
from __future__ import annotations

# Official R4 (unchanged denominators / formula).
R4_TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")

# User-required external evidence set (sci-img / mmmu / ai2d / vqa-text).
# textvqa_val_ocr = LLaVA official TextVQA protocol (OCR prompt).
EXTRA_TASKS = (
    "scienceqa_img",
    "mmmu_val",
    "ai2d",
    "textvqa_val_ocr",
)

# Default full ACC queue = R4 + extras (8 tasks).
ACC_TASKS = R4_TASKS + EXTRA_TASKS

EXTRA_METRIC = {
    "scienceqa_img": "exact_match",
    "mmmu_val": "mmmu_acc",
    "ai2d": "exact_match",
    "textvqa_val_ocr": "exact_match",
}
