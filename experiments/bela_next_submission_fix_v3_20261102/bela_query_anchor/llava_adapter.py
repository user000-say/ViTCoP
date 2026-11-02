"""Minimal, fail-closed LLaVA integration helper for the QAnchor Stage1 arm.

Requires a call site that already owns:
  - native Dual COVER IDs [K] and its full-grid metric [N,D];
  - full-grid CLIP visual hidden states which mm_projector actually expects;
  - QUESTION-ONLY input IDs (without image placeholders, labels or answers);
  - the current model's mm_projector and LLM input embedding module.

This module does NOT patch private LLaVA method signatures automatically.
Cursor MUST locate the real production call site and pass request-local data.
"""
from __future__ import annotations

from typing import Optional
import torch

from .query_anchor import (
    AnchorConfig, AnchorResult, select_question_anchored,
    matched_random_entry_control,
)


@torch.no_grad()
def apply_llava_query_anchor(
    *,
    mode: str,
    native_keep_ids: torch.Tensor,
    dual_metric: torch.Tensor,
    full_vision_features_for_projector: torch.Tensor,
    question_only_ids: torch.Tensor,
    question_content_mask: Optional[torch.Tensor] = None,
    native_saliency: Optional[torch.Tensor] = None,
    mm_projector: torch.nn.Module,
    llm_input_embeddings: torch.nn.Module,
    projected_full_vision: Optional[torch.Tensor] = None,
    config: AnchorConfig = AnchorConfig(),
    random_seed: int = 17,
    diagnostics: bool = False,
) -> AnchorResult:
    if mode not in ("off", "s1_anchor", "matched_random"):
        raise ValueError("invalid QAnchor arm")
    base = native_keep_ids.long().sort().values
    if mode == "off":
        return AnchorResult(base, base, 0, "disabled", [] if diagnostics else None)

    if question_only_ids.ndim != 1:
        raise ValueError("single-sample question_only_ids must be 1-D")
    # Validate the ORIGINAL input, even where a content-word mask would hide tokens.
    if question_only_ids.numel() and torch.any(question_only_ids < 0):
        raise ValueError("question IDs must exclude LLaVA image placeholders")
    if question_content_mask is not None:
        if (question_content_mask.dtype != torch.bool
                or question_content_mask.shape != question_only_ids.shape):
            raise ValueError("question_content_mask must be bool and match question_only_ids")
        question_only_ids = question_only_ids[question_content_mask]
    if question_only_ids.numel() == 0:
        return AnchorResult(base, base, 0, "empty_question", [] if diagnostics else None)
    if full_vision_features_for_projector.ndim != 2:
        raise ValueError("full-grid visual features must be [N,D]")
    if int(full_vision_features_for_projector.shape[0]) != int(dual_metric.shape[0]):
        raise ValueError("dual descriptor/real visual features must share original Patch IDs")

    if projected_full_vision is None:
        # IMPORTANT: this full-grid projector invocation may add computation.
        # Never attribute this overhead to the pre-existing BELA baseline.
        # Match projector param dtype (ACC path is usually fp16; fair-256 bench can hand fp32).
        # ACC wraps mm_projector in _TimedProjector (no .parameters); unwrap if needed.
        proj_mod = getattr(mm_projector, "module", mm_projector)
        try:
            proj_dtype = next(proj_mod.parameters()).dtype
        except Exception:
            proj_dtype = full_vision_features_for_projector.dtype
        features = full_vision_features_for_projector
        if features.dtype != proj_dtype:
            features = features.to(dtype=proj_dtype)
        projected = mm_projector(features)
    else:
        projected = projected_full_vision
    if projected.ndim != 2:
        raise ValueError("projected full visual features must be [N,LLM_hidden]")
    if projected.device != dual_metric.device:
        raise ValueError("Dual metric and projected features must be on the same device; do not silently copy GPU states to CPU")

    # Same device as the LLM embedding table; never use target text.
    question_emb = llm_input_embeddings(question_only_ids.to(
        llm_input_embeddings.weight.device
    ))
    selected = select_question_anchored(
        native_keep_ids=base,
        visual_metric=dual_metric,
        visual_llm_embeds=projected,
        question_embeds=question_emb,
        native_saliency=native_saliency,
        config=config,
        record=diagnostics,
    )
    if mode == "matched_random":
        ids = matched_random_entry_control(
            base, selected.keep_ids, dual_metric.shape[0], seed=random_seed
        )
        return AnchorResult(ids, base, int((~torch.isin(base, ids)).sum()),
                            "matched_random_control", selected.records)
    return selected
