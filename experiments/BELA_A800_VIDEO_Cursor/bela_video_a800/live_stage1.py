"""Stage1-only live helper for LLaVA-Video pooled tokens.

This is NOT full BELA (no RDRIFT / decoder cuts). It only chooses which of the
169 post-pool patches survive into the LLM prompt per frame.

Native keep is L2-norm Top-K on the pooled projected rows, labeled
NATIVE_L2_TOPK — not Dual COVER. QA-COVER residual uses the frozen formula
from bela_video_a800.video_entry when mode=qacover.

BELA_VIDEO_MODE:
  off|dense|unset — caller must leave the official path untouched
  native          — NATIVE_L2_TOPK only
  qacover         — NATIVE_L2_TOPK then frame-wise QA-COVER
  matched_random  — same swap count as QA, random donors
  replay          — gather frozen ids from model._bela_replay_stage1_ids (P0 diagnostic)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from .schedule import budget
from .video_entry import select_framewise


@dataclass
class Stage1Trace:
    mode: str
    tier: str
    frames: int
    patches: int
    keep_per_frame: int
    ids: torch.Tensor  # [F,K]
    swaps: tuple[int, ...]
    reason: str
    selector: str


def _mode() -> str:
    return os.environ.get("BELA_VIDEO_MODE", "off").strip().lower() or "off"


def active() -> bool:
    return _mode() not in ("off", "dense", "0", "false", "no")


def _tier() -> str:
    return os.environ.get("BELA_VIDEO_TIER", "T32").strip().upper()


def native_l2_topk(pooled: torch.Tensor, k: int) -> torch.Tensor:
    """pooled: [F,N,H] → [F,K] ascending original patch ids."""
    if pooled.ndim != 3:
        raise ValueError("pooled must be [F,N,H]")
    f, n, _ = pooled.shape
    if not (0 < k <= n):
        raise ValueError(f"invalid K={k} for N={n}")
    score = pooled.detach().float().norm(dim=-1)  # [F,N]
    # topk largest; then sort ids ascending for a stable gather order
    idx = score.topk(k, dim=-1).indices
    return idx.sort(dim=-1).values.long()


def question_content_embeds(model, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Question-only LLM input embeddings. No image placeholders, no answers."""
    if input_ids is None:
        return None
    from llava.constants import IMAGE_TOKEN_INDEX

    ids = input_ids
    if ids.ndim == 2:
        ids = ids[0]
    ids = ids.detach().long()
    keep = ids[(ids >= 0) & (ids != IMAGE_TOKEN_INDEX)]
    if keep.numel() == 0:
        return None
    emb = model.get_model().embed_tokens(keep.to(model.device))
    return emb


def select_and_gather(
    *,
    model,
    pooled: torch.Tensor,
    input_ids: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Stage1Trace]:
    """Return gathered [F,K,H] and a trace. Caller adds structural newlines."""
    mode = _mode()
    if mode in ("off", "dense"):
        raise RuntimeError("select_and_gather must not run when mode is off")
    tier = _tier()
    f, n, h = pooled.shape
    spec = budget(tier, frames=f, patches_per_frame=n)
    k = spec.framewise_counts[0]

    # P0 diagnostic: replay frozen QA (or any) keep IDs without live scoring.
    if mode == "replay":
        raw = getattr(model, "_bela_replay_stage1_ids", None)
        if raw is None:
            raise RuntimeError("replay mode requires model._bela_replay_stage1_ids")
        ids = torch.as_tensor(raw, device=pooled.device, dtype=torch.long)
        if ids.ndim != 2 or ids.shape != (f, k):
            raise RuntimeError(f"replay ids shape {tuple(ids.shape)} != {(f, k)}")
        swaps = tuple(0 for _ in range(f))
        reason = "replay_frozen_ids"
        selector = "REPLAY"
        gathered = torch.gather(pooled, 1, ids.unsqueeze(-1).expand(-1, -1, h))
        trace = Stage1Trace(
            mode=mode, tier=tier, frames=f, patches=n, keep_per_frame=k,
            ids=ids.detach().cpu(), swaps=swaps, reason=reason, selector=selector,
        )
        return gathered, trace

    native = native_l2_topk(pooled, k)
    control = {
        "native": "off",  # keep native ids inside select_framewise
        "qacover": "qa",
        "matched_random": "matched_random",
    }.get(mode)
    if control is None:
        raise ValueError(f"unknown BELA_VIDEO_MODE={mode}")

    q_emb = None
    if control != "off":
        q_emb = question_content_embeds(model, input_ids)

    # Dual views: same pooled rows twice. This is an explicit Stage1 descriptor
    # stand-in until SigLIP middle/penultimate Dual COVER is wired.
    dual = (pooled, pooled)
    if control == "off" or q_emb is None:
        ids = native
        swaps = tuple(0 for _ in range(f))
        reason = "native_l2_topk" if q_emb is None and mode == "qacover" else "native_l2_topk"
        selector = "NATIVE_L2_TOPK"
    else:
        out = select_framewise(
            tier=tier,
            native_ids=native,
            dual_views=dual,
            projected_full_pooled_vision=pooled,
            question_content_embeddings=q_emb,
            enabled=True,
            control=control,
        )
        ids = out.ids
        swaps = out.swap_count
        reason = ",".join(out.reasons)
        selector = "NATIVE_L2_TOPK+QA_COVER" if mode == "qacover" else "NATIVE_L2_TOPK+MATCHED_RANDOM"

    gathered = torch.gather(
        pooled, 1, ids.unsqueeze(-1).expand(-1, -1, h)
    )
    if gathered.shape != (f, k, h):
        raise RuntimeError(f"gather shape {tuple(gathered.shape)} != {(f,k,h)}")
    trace = Stage1Trace(
        mode=mode, tier=tier, frames=f, patches=n, keep_per_frame=k,
        ids=ids.detach().cpu(), swaps=swaps, reason=reason, selector=selector,
    )
    return gathered, trace
