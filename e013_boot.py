#!/usr/bin/env python3
"""Run e010_eval_job against the E013 runtime tree.

apply_profile resets the selector. This restores the E013 arm after that call,
without changing the QED formula.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))

import e010_common
from e010_common import sha256
from e013_arms import apply_arm_env

e010_common.ROOT = REPO / "logs/e013_p1"
e010_common.RUNTIME = e010_common.ROOT / "code_runtime"

import e009_common

_ORIG = e009_common.apply_profile


def _sync_e013_support() -> None:
    name = "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"
    src = REPO / name
    dst = e010_common.RUNTIME / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or sha256(dst) != sha256(src):
        shutil.copy2(src, dst)


def apply_profile(*args, **kwargs):
    _ORIG(*args, **kwargs)
    arm = os.environ["E013_ARM"]
    meta = apply_arm_env(arm)
    import e006_span

    e006_span.STAGE1_VISUAL = meta["visual"]
    # Formal accuracy: audits off.
    os.environ["E013_AUDIT"] = "False"
    os.environ["E013_STRICT"] = "False"
    os.environ["E012_KV_AUDIT"] = "False"
    os.environ["E012_DIAG"] = "False"


e009_common.apply_profile = apply_profile
e010_common.bind()
_sync_e013_support()

import runpy

runpy.run_path(str(REPO / "e010_eval_job.py"), run_name="__main__")
