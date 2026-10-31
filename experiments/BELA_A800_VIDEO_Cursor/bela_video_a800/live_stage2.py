"""Video Stage2 physical prune helpers for LLaVA-Video / Qwen2.

Frame layout after Stage1: [patch]*K + newline, repeated F times.
Only patch slots are scored/pruned; newlines and all non-visual tokens stay.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .schedule import CUTS_AFTER_LAYERS_1BASED, SCHEDULES, budget


def stage2_mode() -> str:
    return os.environ.get("BELA_VIDEO_STAGE2", "OFF").strip().upper() or "OFF"


def stage2_active() -> bool:
    return stage2_mode() not in ("OFF", "0", "FALSE", "NO", "")


@dataclass
class Stage2State:
    frames: int
    entry_k: int
    patch_ids: torch.Tensor  # [F, K0] original pooled patch ids
    visual_start: int
    events: list = field(default_factory=list)
    ranks_history: Optional[torch.Tensor] = None  # [F, K_alive] last ranks
    cut_idx: int = 0
    physical_layers: list = field(default_factory=list)
    observed_visual_k: list = field(default_factory=list)

    @property
    def stride(self) -> int:
        return self.entry_k + 1  # patches + newline, but entry_k shrinks after cuts

    def current_k(self) -> int:
        return int(self.patch_ids.shape[1])


def targets_for_tier(tier: str) -> tuple[int, int, int]:
    return SCHEDULES[tier.upper()]


def build_keep_mask_framewise(
    scores: torch.Tensor,  # [F, K]
    k_keep: int,
) -> torch.Tensor:
    """Return bool mask [F, K] selecting top-k_keep per frame (stable by id order after)."""
    if scores.ndim != 2:
        raise ValueError("scores must be [F,K]")
    f, k = scores.shape
    if not (0 < k_keep <= k):
        raise ValueError(f"k_keep={k_keep} invalid for K={k}")
    # topk then sort indices ascending to keep spatial order
    idx = scores.topk(k_keep, dim=-1).indices.sort(dim=-1).values
    mask = torch.zeros(f, k, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, idx, True)
    return mask


def q_attn_scores(attn_avg: torch.Tensor, visual_start: int, visual_end: int) -> torch.Tensor:
    """Mean text→visual attention. attn_avg: [seq, seq] (already head-averaged)."""
    text = attn_avg[visual_end:, visual_start:visual_end]
    if text.numel() == 0:
        return torch.ones(visual_end - visual_start, device=attn_avg.device, dtype=torch.float32)
    return text.mean(dim=0).float()


def q_attn_scores_from_hidden(
    hidden_states: torch.Tensor, visual_start: int, visual_end: int
) -> torch.Tensor:
    """Question/text → visual scores via layer hidden states (batch=1).

    Avoids Qwen2 SDPA→eager `output_attentions` path, which trips device asserts
    on this Transformers 4.40 + Qwen2 stack. Scores are mean softmax attention
    from post-visual tokens onto the visual span (same aggregation spirit as
    STAR TEXT_AGG_MODE=average_all).
    """
    assert hidden_states.shape[0] == 1
    hs = hidden_states[0].float()
    vis = hs[visual_start:visual_end]  # [Nv, D]
    txt = hs[visual_end:]  # [Nt, D]
    if txt.numel() == 0:
        return torch.ones(visual_end - visual_start, device=hs.device, dtype=torch.float32)
    # [Nt, Nv]
    logits = txt @ vis.transpose(0, 1) / max(vis.shape[-1] ** 0.5, 1.0)
    attn = torch.softmax(logits, dim=-1)
    return attn.mean(dim=0)


def ranks_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Higher score → better (lower) rank. scores [F,K] → ranks [F,K] in 0..K-1."""
    # argsort descending → rank
    order = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(scores.shape[-1], device=scores.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks.float()


def rdrift_scores(now_scores: torch.Tensor, old_ranks: torch.Tensor) -> torch.Tensor:
    """0.6*rank(now)+0.4*(rank(now)-rank(old)); lower is better → invert for topk."""
    now_ranks = ranks_from_scores(now_scores)
    combo = 0.6 * now_ranks + 0.4 * (now_ranks - old_ranks)
    # convert to keep-preference (higher better)
    return -combo


def visual_region_keep_indices(
    state: Stage2State,
    patch_mask: torch.Tensor,  # [F, K]
) -> torch.Tensor:
    """Absolute sequence indices to keep inside the visual block (patches+newlines)."""
    mask = patch_mask.detach().to("cpu").bool()
    f, k = mask.shape
    stride = k + 1
    keeps = []
    for fi in range(f):
        base = fi * stride
        for pi in range(k):
            if mask[fi, pi].item():
                keeps.append(base + pi)
        keeps.append(base + k)  # newline
    return torch.tensor(keeps, dtype=torch.long)


def gather_seq(x: torch.Tensor, keep_abs: torch.Tensor, visual_start: int, visual_end: int, dim: int) -> torch.Tensor:
    """Gather along sequence dim, replacing visual_start:visual_end with keep_abs relative picks."""
    # keep_abs are offsets within visual region
    device = x.device
    pre = torch.arange(visual_start, device=device)
    post = torch.arange(visual_end, x.shape[dim], device=device)
    abs_idx = torch.cat([pre, visual_start + keep_abs, post], dim=0)
    return x.index_select(dim, abs_idx)


def prune_cache(past_key_values, keep_abs: torch.Tensor, visual_start: int, visual_end: int, layers_filled: int):
    """Physically shorten KV for layers already written (0..layers_filled-1)."""
    if past_key_values is None or layers_filled <= 0:
        return past_key_values
    from transformers.cache_utils import DynamicCache, Cache

    if isinstance(past_key_values, DynamicCache):
        new = DynamicCache()
        for i in range(min(layers_filled, len(past_key_values.key_cache))):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            # k,v: [B, heads, seq, dim]
            seq = k.shape[2]
            pre = torch.arange(visual_start, device=k.device)
            post = torch.arange(visual_end, seq, device=k.device)
            abs_idx = torch.cat([pre, visual_start + keep_abs.to(k.device), post], dim=0)
            new.update(k.index_select(2, abs_idx), v.index_select(2, abs_idx), i)
        return new
    # legacy tuple list
    out = []
    for i, layer in enumerate(past_key_values):
        if i >= layers_filled or layer is None:
            out.append(layer)
            continue
        k, v = layer[0], layer[1]
        seq = k.shape[2]
        pre = torch.arange(visual_start, device=k.device)
        post = torch.arange(visual_end, seq, device=k.device)
        abs_idx = torch.cat([pre, visual_start + keep_abs.to(k.device), post], dim=0)
        out.append((k.index_select(2, abs_idx), v.index_select(2, abs_idx)) + tuple(layer[2:]))
    return out


def select_and_prune(
    *,
    state: Stage2State,
    hidden_states: torch.Tensor,
    attn_avg: Optional[torch.Tensor],  # [seq,seq] or None → use hidden proxy
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values,
    layer_1based: int,
    k_target: int,
    selector: str,
    layers_filled: int,
):
    """Apply one physical cut. Returns updated tensors + state."""
    k = state.current_k()
    f = state.frames
    visual_len = f * (k + 1)
    visual_start = state.visual_start
    visual_end = visual_start + visual_len
    if hidden_states.shape[1] < visual_end:
        raise RuntimeError(
            f"seq={hidden_states.shape[1]} < visual_end={visual_end}; Stage2 layout mismatch"
        )

    seq_len = int(hidden_states.shape[1])
    if not (0 <= visual_start < visual_end <= seq_len):
        raise RuntimeError(
            f"visual span invalid start={visual_start} end={visual_end} seq={seq_len} "
            f"F={f} K={k} expected_visual={visual_len}"
        )
    # P1 micro-opt: keep scoring on GPU (same hidden/attn formulas; no D2H).
    # P3 same-layer Q/K replacement is a separate semantic change — not here.
    device = hidden_states.device
    if attn_avg is not None:
        if int(attn_avg.shape[0]) != seq_len or int(attn_avg.shape[1]) != seq_len:
            raise RuntimeError(
                f"attn_avg shape {tuple(attn_avg.shape)} != seq {seq_len}"
            )
        flat_scores = q_attn_scores(attn_avg.float(), visual_start, visual_end)
    else:
        flat_scores = q_attn_scores_from_hidden(
            hidden_states.detach().float(), visual_start, visual_end
        )
    if int(flat_scores.numel()) != visual_len:
        raise RuntimeError(
            f"flat_scores {int(flat_scores.numel())} != visual_len {visual_len}"
        )
    scores_blk = flat_scores.view(f, k + 1)
    patch_scores = scores_blk[:, :k].contiguous()

    if selector == "RDRIFT":
        if state.ranks_history is None or state.ranks_history.shape != patch_scores.shape:
            old = ranks_from_scores(patch_scores)
        else:
            old = state.ranks_history.to(device=patch_scores.device)
        pref = rdrift_scores(patch_scores, old)
    else:
        pref = patch_scores

    mask = build_keep_mask_framewise(pref, k_target)
    # Vectorized keep indices inside visual region (patches + newlines).
    # layout per frame: [p0..p{k-1}, newline]; stride = k+1
    fi = torch.arange(f, device=device).view(f, 1)
    pi = torch.arange(k, device=device).view(1, k)
    patch_abs = (fi * (k + 1) + pi)[mask]  # kept patch offsets
    nl_abs = torch.arange(f, device=device) * (k + 1) + k
    # Interleave by frame: for each frame, kept patches (sorted by pi) then newline.
    # Rebuild via list-free: sort key = frame * (k+1) + local
    keep_rel_dev = torch.cat([patch_abs, nl_abs], dim=0).sort().values
    # update patch ids + ranks on surviving slots (for next RDRIFT cut)
    now_ranks = ranks_from_scores(patch_scores)
    # gather surviving ids: [F,k_target]
    # mask [F,K] → take top positions already sorted ascending in build_keep_mask
    idx = mask.to(torch.long).argsort(dim=-1, descending=True)[:, :k_target]
    idx = idx.sort(dim=-1).values
    patch_ids_dev = state.patch_ids.to(device=device)
    if patch_ids_dev.dtype != torch.long:
        patch_ids_dev = patch_ids_dev.long()
    state.patch_ids = patch_ids_dev.gather(1, idx).detach().cpu()
    state.ranks_history = now_ranks.gather(1, idx).detach()

    hs = gather_seq(hidden_states, keep_rel_dev, visual_start, visual_end, dim=1)
    # Re-index positions 0..L-1 after physical prune. Keeping sparse original
    # position ids OOBs Qwen2 RoPE cos/sin tables sized by current kv_seq_len.
    if position_ids is not None:
        new_len = hs.shape[1]
        position_ids = torch.arange(new_len, device=hs.device, dtype=position_ids.dtype).unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 4:
        am = attention_mask
        am = gather_seq(am, keep_rel_dev, visual_start, visual_end, dim=3)
        am = gather_seq(am, keep_rel_dev, visual_start, visual_end, dim=2)
        attention_mask = am
    elif attention_mask is not None and attention_mask.dim() == 2:
        attention_mask = gather_seq(attention_mask, keep_rel_dev, visual_start, visual_end, dim=1)

    past_key_values = prune_cache(past_key_values, keep_rel_dev, visual_start, visual_end, layers_filled)

    before = visual_len
    after = f * (k_target + 1)
    state.events.append(
        {
            "layer_1based": layer_1based,
            "selector": selector,
            "visual_before": before,
            "visual_after": after,
            "k_before": k,
            "k_after": k_target,
        }
    )
    state.physical_layers.append(layer_1based)
    state.cut_idx += 1
    return hs, position_ids, attention_mask, past_key_values, state


def cut_plan(tier: str) -> list[tuple[int, int, str]]:
    """List of (layer_1based, k_after, selector) for the two cuts."""
    _, k1, k2 = targets_for_tier(tier)
    mode = stage2_mode()
    c1, c2 = CUTS_AFTER_LAYERS_1BASED
    if mode == "Q_ATTN_ALL":
        return [(c1, k1, "Q_ATTN"), (c2, k2, "Q_ATTN")]
    if mode == "RDRIFT_FIRST":
        return [(c1, k1, "RDRIFT"), (c2, k2, "Q_ATTN")]
    raise ValueError(f"unknown BELA_VIDEO_STAGE2={mode}")
