"""Reference frame-wise QA-COVER for video. Not a production decoder patch.

Caller must supply actual 169 *post-pooling* tokens per frame, not raw SigLIP
27x27=729 prepool features, and a K-frame native Dual COVER selection.
The same exact pretrained visual feature rows are gathered; no merging.
No global (64*169)^2 similarity is ever materialized.

Output IDs are (frame_id, pooled_patch_id) identities in the *visual-only*
sequence, NOT positions in the full LLM text/video/time prompt.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .qa_core import (
    AnchorConfig,
    select_question_anchored,
    matched_random_entry_control,
)
from .schedule import budget, FRAME_PATCHES


@dataclass
class FramewiseSelection:
    ids: torch.Tensor                    # [F,K] pooled patch IDs per frame
    original_ids: torch.Tensor           # [F,K] native Dual COVER IDs
    swap_count: tuple[int, ...]          # per frame
    reasons: tuple[str, ...]
    # Flat indices within visual-only sequence; not decoder prompt offsets.
    flattened_visual_only_ids: torch.Tensor


def query_proxy(
    projected_full_pooled_vision: torch.Tensor,  # [F,169,H]
    question_content_embeddings: torch.Tensor, # [Q,H]
) -> torch.Tensor:
    if projected_full_pooled_vision.ndim != 3:
        raise ValueError("visual projector output must be [F,N,H]")
    if question_content_embeddings.ndim != 2:
        raise ValueError("question-only embeddings must be [Q,H]")
    if (
        projected_full_pooled_vision.shape[-1]
        != question_content_embeddings.shape[-1]
    ):
        raise ValueError("visual/text LLM embedding dimensions differ")
    if question_content_embeddings.shape[0] == 0:
        return torch.zeros(
            projected_full_pooled_vision.shape[:2],
            device=projected_full_pooled_vision.device,
            dtype=torch.float32,
        )
    if (not torch.isfinite(projected_full_pooled_vision).all()
            or not torch.isfinite(question_content_embeddings).all()):
        raise ValueError("non-finite features")
    t = question_content_embeddings.detach().float().to(
        projected_full_pooled_vision.device
    ).mean(0, keepdim=True)
    t = F.normalize(t, dim=-1, eps=1e-12)
    v = F.normalize(
        projected_full_pooled_vision.detach().float(), dim=-1, eps=1e-12
    )
    return (v @ t.T).squeeze(-1)


def select_framewise(
    *,
    tier: str,
    native_ids: torch.Tensor,                 # [F,K] original pooled-patch IDs
    dual_views: Sequence[torch.Tensor],      # each [F,N,Dv]; aligned to pooled unit
    raw_query_scores: Optional[torch.Tensor] = None,  # [F,N]
    projected_full_pooled_vision: Optional[torch.Tensor] = None,
    question_content_embeddings: Optional[torch.Tensor] = None,
    enabled: bool = True,
    control: str = "qa",  # qa, off, matched_random
    seed: int = 2027,
) -> FramewiseSelection:
    if control not in ("qa", "off", "matched_random"):
        raise ValueError("unknown control")
    if native_ids.ndim != 2 or native_ids.dtype not in (
        torch.int64, torch.int32
    ):
        raise ValueError("native_ids must be integer [F,K]")
    f, k = native_ids.shape
    if not dual_views:
        raise ValueError("Dual COVER views must be provided")
    n = int(dual_views[0].shape[1])
    budget_spec = budget(tier, frames=f, patches_per_frame=n)
    if k != budget_spec.framewise_counts[0]:
        raise ValueError("native K does not match the fixed video tier")
    dev = native_ids.device
    for view in dual_views:
        if view.ndim != 3 or tuple(view.shape[:2]) != (f,n):
            raise ValueError("each dual view must be aligned [F,N,D]")
        if view.device != dev or not torch.isfinite(view).all():
            raise ValueError("visual features invalid or on different device")
    if native_ids.numel() and (
        native_ids.min() < 0 or native_ids.max() >= n
    ):
        raise ValueError("original pooled IDs out of range")
    if any(row.unique().numel() != k for row in native_ids):
        raise ValueError("duplicate original pooled IDs in frame")
    if raw_query_scores is not None and (
        projected_full_pooled_vision is not None
        or question_content_embeddings is not None
    ):
        raise ValueError("Use precomputed scores OR projected + question embeddings")
    if raw_query_scores is None and (
        projected_full_pooled_vision is not None
        and question_content_embeddings is not None
    ):
        raw_query_scores=query_proxy(
            projected_full_pooled_vision, question_content_embeddings
        )
    if raw_query_scores is not None and (
        raw_query_scores.shape != (f,n)
        or not torch.isfinite(raw_query_scores).all()
    ):
        raise ValueError("raw_query_scores must be finite [F,N]")
    if control == "matched_random" and raw_query_scores is None:
        raise ValueError("random control requires QA selection proposal to match swaps")
    output, counts, reasons = [], [], []
    # Fix swap fraction = 8/64 = 12.5% of initial entry cardinality.
    max_swaps = 8 if tier == "T32" else 4
    for frame in range(f):
        original = native_ids[frame].long().sort().values
        if not enabled or control == "off" or raw_query_scores is None:
            selected=original
            used=0
            reason="native_noop"
        else:
            candidate=select_question_anchored(
                native_keep_ids=original,
                visual_metric=tuple(view[frame] for view in dual_views),
                raw_query_scores=raw_query_scores[frame],
                config=AnchorConfig(max_swaps=max_swaps),
            )
            used=candidate.swaps
            selected=candidate.keep_ids
            reason=candidate.reason
            if control=="matched_random":
                selected=matched_random_entry_control(
                    original, candidate.keep_ids, n_patches=n,
                    seed=seed+frame,
                )
                reason="matched_random_donors"
        if selected.numel() != k or selected.unique().numel() != k:
            raise RuntimeError("frame-wise selection violated fixed K")
        output.append(selected.to(device=dev))
        counts.append(used)
        reasons.append(reason)
    out=torch.stack(output,dim=0)
    visual_flat_ids=(
        out + torch.arange(f,device=dev,dtype=torch.long)[:,None]*n
    ).reshape(-1)
    if visual_flat_ids.unique().numel() != f*k:
        raise RuntimeError("global pooled visual-only IDs duplicated")
    return FramewiseSelection(
        ids=out, original_ids=native_ids.long().sort(dim=-1).values,
        swap_count=tuple(counts), reasons=tuple(reasons),
        flattened_visual_only_ids=visual_flat_ids,
    )
