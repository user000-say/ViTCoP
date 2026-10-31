"""SigLIP Dual-COVER *adaptation* for 169 pooled video patches (P2).

NOT byte-identical to image E013 COVER (GRID=24 / CLS scores). See
docs/VIDEO_DUAL_COVER_MAPPING_AUDIT.md.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

GRID = 13  # 13*13=169 after pool
CELLS = 1  # 13 not divisible by 4; single-cell novelty = 1 - cos to anchor mean
N_PATCH = GRID * GRID  # 169


def pool_tokens_27_to_13(tokens_729: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """tokens: [F,729,D] or [729,D] → [F,169,D] / [169,D] via avg pool."""
    single = tokens_729.ndim == 2
    if single:
        tokens_729 = tokens_729.unsqueeze(0)
    f, n, d = tokens_729.shape
    if n != 729:
        raise ValueError(f"expected 729 pre-pool tokens, got {n}")
    x = tokens_729.view(f, 27, 27, d).permute(0, 3, 1, 2).contiguous()
    x = F.avg_pool2d(x, stride)
    x = x.permute(0, 2, 3, 1).contiguous().view(f, -1, d)
    if x.shape[1] != N_PATCH:
        raise RuntimeError(f"pooled N={x.shape[1]} != {N_PATCH}")
    return x[0] if single else x


def _relative_coverage_169(
    tokens: torch.Tensor, scores: torch.Tensor, anchor_ids: torch.Tensor, residual_ids: torch.Tensor
) -> torch.Tensor:
    """Port of E013 cell novelty with GRID=13, CELLS=1 (global prototype)."""
    features = F.normalize(tokens.float(), p=2, dim=-1, eps=1e-8)
    if anchor_ids.numel() == 0:
        return scores.float().index_select(0, residual_ids)
    proto = F.normalize(features.index_select(0, anchor_ids).mean(0, keepdim=True), p=2, dim=-1, eps=1e-8)
    cos = (features.index_select(0, residual_ids) * proto).sum(-1)
    novelty = 1.0 - cos.clamp(0, 1)
    return scores.float().index_select(0, residual_ids) * novelty


def cover_replace_169(
    metric_features: torch.Tensor,  # [169,D]
    scores: torch.Tensor,  # [169]
    k: int = 64,
    slots: int = 32,
) -> torch.Tensor:
    """REPLACE COVER → K unique IDs sorted ascending."""
    if metric_features.shape[0] != N_PATCH or scores.numel() != N_PATCH:
        raise ValueError("cover_replace_169 expects N=169")
    if not (0 < k <= N_PATCH) or slots < 0 or slots > k:
        raise ValueError(f"invalid k/slots {k}/{slots}")
    device = metric_features.device
    scores = scores.float().to(device)
    if slots == 0:
        return scores.topk(k, largest=True, sorted=False).indices.sort().values
    anchors_k = k - slots
    anchors = (
        scores.topk(anchors_k, largest=True, sorted=False).indices
        if anchors_k > 0
        else scores.new_empty(0, dtype=torch.long)
    )
    # Residual excludes CLS-top-K (same spirit as E013 excluding Top core_k).
    topk_mask = torch.zeros(N_PATCH, dtype=torch.bool, device=device)
    topk_mask[scores.topk(k, largest=True, sorted=False).indices] = True
    residual = torch.where(~topk_mask)[0]
    util = _relative_coverage_169(metric_features, scores, anchors, residual)
    n_pick = min(slots, int(residual.numel()))
    pick = residual[util.topk(n_pick, largest=True, sorted=False).indices]
    keep = torch.cat([anchors, pick], dim=0)
    if keep.unique().numel() != k:
        raise RuntimeError(f"COVER keep unique={keep.unique().numel()} != K={k}")
    return keep.sort().values


def dual_cover_framewise(
    mid_729: torch.Tensor,  # [F,729,D]
    pen_729: torch.Tensor,  # [F,729,D]
    k: int = 64,
    slots: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return keep_ids [F,K], mid_169, pen_169."""
    mid = pool_tokens_27_to_13(mid_729)
    pen = pool_tokens_27_to_13(pen_729)
    f = mid.shape[0]
    scores = pen.float().norm(dim=-1)  # adaptation saliency
    # Dual metric: concat mid||pen then use for coverage geometry
    metric = torch.cat([mid.float(), pen.float()], dim=-1)
    keeps = []
    for i in range(f):
        keeps.append(cover_replace_169(metric[i], scores[i], k=k, slots=slots))
    return torch.stack(keeps, dim=0), mid, pen
