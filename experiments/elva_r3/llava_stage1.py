"""Stage1 selection on the CLIP forward LLaVA already ran.

Payload stays the penultimate patch rows that the existing projector consumes.
Saliency uses this repository's live rule: sum of attention heads, CLS to patches.
That is the historical baseline, and it is not the package mean-then-L1 score.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from elva.cover_binding import NativeCover, file_sha, select_checked
from elva.low_budget_cover import POLICY, callback as pcov_callback
from native_cover import legacy_callback

PKG = Path("/home/xiangyue/ViTCoP/experiments/elva_r3")
E013 = Path("/home/xiangyue/ViTCoP/LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py")
_COVERS = None


def _covers():
    global _COVERS
    if _COVERS is not None:
        return _COVERS
    legacy_files = {
        str(Path("/home/xiangyue/ViTCoP/experiments/elva_r3/native_cover.py").resolve()): file_sha(
            "/home/xiangyue/ViTCoP/experiments/elva_r3/native_cover.py"),
        str(E013.resolve()): file_sha(E013),
    }
    pcov = PKG / "elva/low_budget_cover.py"
    _COVERS = {
        "legacy_native": NativeCover(legacy_callback, dict(
            family="legacy_native", entrypoint="native_cover:legacy_callback",
            supported_k=[172], replacement_slots={"172": 32}, source_files=legacy_files)),
        "protected_coverage": NativeCover(pcov_callback, dict(
            family="protected_coverage", entrypoint="elva.low_budget_cover:callback",
            supported_k=[60], replacement_slots={"60": 12}, allowed_slots=[6, 12, 18],
            coverage_policy=POLICY, source_files={str(pcov.resolve()): file_sha(pcov)})),
    }
    return _COVERS


def _sum_cls(attentions, index):
    # attentions[index]: [1, heads, 577, 577] -> [576]
    return attentions[index][0, :, 0, 1:].sum(0).float()


def select_existing_forward(image_forward_outs, tokens_0):
    config = json.loads(Path(os.environ["ELVA_R3_CONFIG"]).read_text())
    hidden = image_forward_outs.hidden_states
    pen = hidden[-2][0, 1:]
    if pen.shape != tokens_0.shape or pen.device != tokens_0.device or not torch.equal(pen, tokens_0):
        raise RuntimeError("ELVA payload is not the pen rows already used as projector input")
    stage = config["method"]["stage1"]
    guide = stage["guide"]
    beta = float(stage.get("beta") or 0.0)
    score_guide = stage.get("score_guide", "same")
    sg = guide if score_guide == "same" else score_guide
    need_early = (guide == "l9" or (guide == "dual" and beta > 0)
                  or sg == "l9" or (sg == "dual" and beta > 0))
    early = hidden[9][0, 1:] if need_early else None
    pen_scores = _sum_cls(image_forward_outs.attentions, -2)
    early_scores = _sum_cls(image_forward_outs.attentions, 8) if need_early else None
    ids = torch.arange(pen.shape[0], device=pen.device, dtype=torch.long)
    family = stage["cover_family"]
    cover = _covers()[family] if family != "none" else None
    selected = select_checked(
        pen, pen_scores, ids, config,
        early=early, early_scores=early_scores,
        early_ids=ids if need_early else None,
        native_cover=cover, backend="torch")
    keep = selected["keep"]
    if not torch.equal(selected["payload"], pen.index_select(0, keep)):
        raise RuntimeError("selected payload left the captured pen rows")
    audit = selected["stage1_audit"]
    audit_dir = os.environ.get("ELVA_R3_AUDIT_DIR", "")
    if audit_dir:
        path = Path(audit_dir) / "stage1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(ids=keep.detach().cpu().tolist(), audit=audit,
                   score_rule="sum_heads_cls_to_patch")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return keep, audit
