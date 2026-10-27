#!/usr/bin/env python3
"""Run e010_eval_job against the E014 runtime tree.

apply_profile resets the selector. This restores the E014 arm after that call.
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
from e014_arms import apply_arm_env

e010_common.ROOT = REPO / "logs/e014_aee"
e010_common.RUNTIME = e010_common.ROOT / "code_runtime"

import e009_common

_ORIG = e009_common.apply_profile

_SYNC_FILES = (
    "LLaVA/llava/model/language_model/vitcop_modeling_llama.py",
    "LLaVA/llava/model/vitcop_llava_arch.py",
    "LLaVA/llava/model/multimodal_encoder/clip_encoder.py",
    "LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py",
    "LLaVA/llava/model/multimodal_encoder/e014_config.py",
    "LLaVA/llava/model/multimodal_encoder/e014_stage1_support.py",
    "LLaVA/llava/model/multimodal_encoder/e015_regional_config.py",
    "LLaVA/llava/model/multimodal_encoder/e015_regional_support.py",
    "LLaVA/llava/model/multimodal_encoder/e015_stage1_support.py",
    "LLaVA/llava/model/multimodal_encoder/e016_stage1_support.py",
    "LLaVA/llava/model/e018_contract.py",
    "LLaVA/llava/model/e018_counterfactual.py",
    "LLaVA/llava/model/e018_qed_rs_diagnostic.py",
    "LLaVA/llava/model/e018_stage2_support.py",
)


def _sync_e014_runtime() -> None:
    for name in _SYNC_FILES:
        src = REPO / name
        dst = e010_common.RUNTIME / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise SystemExit(f"missing source {src}")
        if not dst.exists() or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)


def apply_profile(*args, **kwargs):
    _ORIG(*args, **kwargs)
    arm = os.environ["E014_ARM"]
    meta = apply_arm_env(arm)
    if os.environ.get("ELVA_R3_CONFIG"):
        import sys
        from pathlib import Path as _ElvaPath
        _elva = str(_ElvaPath("/home/xiangyue/ViTCoP/experiments/elva_r3"))
        if _elva not in sys.path:
            sys.path.insert(0, _elva)
        from apply_llava import overlay
        meta = overlay(meta)
    import e006_span

    e006_span.STAGE1_VISUAL = meta["visual"]
    os.environ["E014_AUDIT"] = "False"
    os.environ["E012_KV_AUDIT"] = "False"
    os.environ["E012_DIAG"] = "False"
    os.environ["E013_ENABLE"] = "False"


e009_common.apply_profile = apply_profile
e010_common.bind()
_sync_e014_runtime()

import runpy

runpy.run_path(str(REPO / "e010_eval_job.py"), run_name="__main__")
