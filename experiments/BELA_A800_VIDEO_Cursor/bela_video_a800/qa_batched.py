"""Exact-equivalent batched QA-COVER for F frames (video P1).

Same formulas/thresholds/pools/tie-break as qa_core.select_question_anchored,
but one Gram per frame-batch and B parallel GPU rounds (no F×B Python/.item hot path).

When dual views are identical (current video proxy dual=(pooled,pooled)), compute
Gram once. Real dual-depth later must re-enable two-view fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .qa_core import AnchorConfig


@dataclass
class BatchedQAResult:
    keep_ids: torch.Tensor  # [F,K] sorted
    native_ids: torch.Tensor  # [F,K]
    swap_count: Tuple[int, ...]
    reasons: Tuple[str, ...]


def _normalize_views(views: Sequence[torch.Tensor]) -> list:
    out = []
    for v in views:
        if v.ndim != 3:
            raise ValueError("each view must be [F,N,D]")
        out.append(F.normalize(v.detach().float(), dim=-1, eps=1e-12))
    return out


def _gram(views: Sequence[torch.Tensor]) -> torch.Tensor:
    """[F,N,N] fused similarity. Identical views → single bmm."""
    norms = _normalize_views(views)
    if len(norms) == 1:
        v = norms[0]
        return (torch.bmm(v, v.transpose(1, 2))).clamp(0.0, 1.0)
    # Exact mean of per-view grams (matches qa_core stack-mean).
    grams = [(torch.bmm(v, v.transpose(1, 2))).clamp(0.0, 1.0) for v in norms]
    # Fast path: if all equal to first, skip extra work (video proxy).
    if all(torch.equal(g, grams[0]) for g in grams[1:]):
        return grams[0]
    return torch.stack(grams, dim=0).mean(dim=0)


@torch.no_grad()
def select_question_anchored_batched(
    *,
    native_keep_ids: torch.Tensor,  # [F,K]
    visual_metric: Sequence[torch.Tensor],  # each [F,N,D]
    raw_query_scores: torch.Tensor,  # [F,N]
    config: AnchorConfig = AnchorConfig(),
) -> BatchedQAResult:
    if native_keep_ids.ndim != 2 or native_keep_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("native_keep_ids must be long [F,K]")
    if raw_query_scores.ndim != 2:
        raise ValueError("raw_query_scores must be [F,N]")
    if not visual_metric:
        raise ValueError("visual_metric required")
    f, k = native_keep_ids.shape
    n = int(visual_metric[0].shape[1])
    device = native_keep_ids.device
    for view in visual_metric:
        if view.shape[:2] != (f, n):
            raise ValueError("visual views must align [F,N,*]")
    if raw_query_scores.shape != (f, n):
        raise ValueError("raw_query_scores shape mismatch")
    if config.saliency_guard_slots:
        raise NotImplementedError("batched path: saliency_guard_slots must be 0 (production video default)")

    native = native_keep_ids.long().to(device).sort(dim=-1).values
    keep = native.clone()
    raw = raw_query_scores.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(raw).all():
        raise ValueError("non-finite query scores")

    # Per-frame flat-query gate (same threshold as float(std) < min_raw_std).
    std = raw.std(dim=-1, unbiased=False)  # [F]
    active = (k > 0) & (k < n) & (config.max_swaps > 0) & (std >= config.min_raw_std)
    # q scores; inactive frames get zeros (unused).
    med = raw.median(dim=-1, keepdim=True).values
    q = torch.sigmoid(((raw - med) / (2.0 * std.clamp_min(1e-12).unsqueeze(-1))).clamp(-8, 8))

    sim = _gram(visual_metric)  # [F,N,N]
    max_steps = min(config.max_swaps, k, n - k)
    swaps = torch.zeros(f, dtype=torch.long, device=device)
    donor_pool = int(config.donor_pool)
    drop_pool = int(config.drop_pool)

    for _ in range(max_steps):
        if not bool(active.any()):
            break
        # active mask [F,N]
        active_patches = torch.zeros(f, n, dtype=torch.bool, device=device)
        active_patches.scatter_(1, keep, True)
        outside = ~active_patches

        # Donor priority over outside; mask inside to -inf
        # novelty = 1 - max_sim(outside, keep)
        # Gather sim[:, outside, keep] via masked amax
        # For each frame: sim[f, i, keep[f]] max over keep
        keep_exp = keep.unsqueeze(1).expand(f, n, k)
        sim_to_keep = sim.gather(2, keep_exp)  # [F,N,K]
        max_to_keep = sim_to_keep.amax(dim=-1)  # [F,N]
        novelty = 1.0 - max_to_keep
        cand_pri = q * novelty
        cand_pri = cand_pri.masked_fill(~outside, float("-inf"))
        # Stable argsort matches qa_core (not unstable topk) for tie-breaks.
        d_actual = min(donor_pool, n)
        donor_order = torch.argsort(cand_pri, dim=-1, descending=True, stable=True)
        donor_idx = donor_order[:, :d_actual]
        donor_vals = cand_pri.gather(1, donor_idx)

        # Drop priority among keep: in_sim[f,a,b] = sim[f, keep[f,a], keep[f,b]]
        bidx = torch.arange(f, device=device)[:, None, None]
        in_sim = sim[bidx, keep.unsqueeze(2), keep.unsqueeze(1)].clone()  # [F,K,K]
        eye = torch.eye(k, device=device, dtype=torch.bool).unsqueeze(0)
        in_sim = in_sim.masked_fill(eye, 0.0)
        redundant = in_sim.amax(dim=-1) if k > 1 else torch.zeros(f, k, device=device, dtype=q.dtype)
        drop_pri = (1.0 - q.gather(1, keep)) + 0.5 * redundant
        drop_pri = drop_pri.masked_fill(~active.unsqueeze(1), float("-inf"))
        r_actual = min(drop_pool, k)
        drop_order = torch.argsort(drop_pri, dim=-1, descending=True, stable=True)
        drop_pos = drop_order[:, :r_actual]
        drop_vals = drop_pri.gather(1, drop_pos)
        drops = keep.gather(1, drop_pos)  # [F,R]

        # Coverage top1/top2 over keep (facility location) — matches qa_core
        top1, argtop = sim_to_keep.max(dim=-1)  # [F,N], arg in 0..K-1
        if k > 1:
            masked = sim_to_keep.clone()
            masked.scatter_(-1, argtop.unsqueeze(-1), -1.0)
            top2 = masked.max(dim=-1).values.clamp_min(0)
        else:
            top2 = torch.zeros_like(top1)
        # without[f,r,n] when dropping drop_pos[f,r]
        # qa_core: without = where(argtop == drop_positions, top2, top1) with shape [drop,N]
        without = torch.where(
            argtop.unsqueeze(1) == drop_pos.unsqueeze(2),  # [F,R,N]
            top2.unsqueeze(1),
            top1.unsqueeze(1),
        )
        donor_sim = sim.gather(2, donor_idx.unsqueeze(1).expand(f, n, d_actual)).permute(0, 2, 1)  # [F,D,N]
        current_cov = top1.mean(dim=-1)  # [F]
        new_cov = torch.maximum(
            without.unsqueeze(2),  # [F,R,1,N]
            donor_sim.unsqueeze(1),  # [F,1,D,N]
        ).mean(dim=-1)  # [F,R,D]
        delta_cov = new_cov - current_cov.view(f, 1, 1)

        q_donors = q.gather(1, donor_idx)  # [F,D]
        q_drops = q.gather(1, drops)  # [F,R]
        query_gain = q_donors.unsqueeze(1) - q_drops.unsqueeze(2)  # [F,R,D]
        donor_novelty = 1.0 - max_to_keep.gather(1, donor_idx)  # [F,D]

        # Valid donors: finite cand_pri (was outside)
        donor_ok = torch.isfinite(donor_vals)  # [F,D]
        drop_ok = torch.isfinite(drop_vals)  # [F,R]
        ok = (
            (query_gain >= config.min_query_gain)
            & (donor_novelty.unsqueeze(1) >= config.min_novelty)
            & (delta_cov >= -config.max_coverage_loss)
            & donor_ok.unsqueeze(1)
            & drop_ok.unsqueeze(2)
            & active.view(f, 1, 1)
        )
        objective = (
            query_gain * donor_novelty.unsqueeze(1)
            + config.coverage_weight * delta_cov
        ).masked_fill(~ok, float("-inf"))

        flat = objective.reshape(f, -1)  # [F, R*D] row-major: drop major, donor minor
        best_flat = flat.argmax(dim=-1)  # [F]
        best_val = flat.gather(1, best_flat.unsqueeze(1)).squeeze(1)
        do_swap = active & torch.isfinite(best_val)
        # Map flat index → (di, dj)
        di = torch.div(best_flat, d_actual, rounding_mode="floor")
        dj = best_flat - di * d_actual
        u = drops.gather(1, di.unsqueeze(1)).squeeze(1)  # [F]
        v = donor_idx.gather(1, dj.unsqueeze(1)).squeeze(1)

        # Apply swaps with identity for inactive: keep_new = sort((keep \ {u}) ∪ {v})
        # Vectorized: replace first occurrence of u with v where do_swap
        keep_new = keep.clone()
        # mask positions equal to u
        is_u = keep_new == u.unsqueeze(1)
        # Only first match: cumsum trick
        first_u = is_u & (is_u.cumsum(dim=-1) == 1)
        repl = torch.where(first_u, v.unsqueeze(1).expand_as(keep_new), keep_new)
        keep_new = torch.where(do_swap.unsqueeze(1), repl, keep)
        keep = keep_new.sort(dim=-1).values
        swaps = swaps + do_swap.long()
        # Frames that failed this round stop further swaps (match per-frame break).
        active = active & do_swap

    # Reasons / counts on CPU only at end (outside hot timed inner logic OK for trace)
    swap_list = tuple(int(x) for x in swaps.detach().cpu().tolist())
    reasons = []
    for i in range(f):
        if not bool((std[i] >= config.min_raw_std).detach().cpu()):
            reasons.append("unreliable_flat_query")
        elif swap_list[i] > 0:
            reasons.append("swapped")
        else:
            reasons.append("no_safe_query_anchor")
    return BatchedQAResult(
        keep_ids=keep,
        native_ids=native,
        swap_count=swap_list,
        reasons=tuple(reasons),
    )
