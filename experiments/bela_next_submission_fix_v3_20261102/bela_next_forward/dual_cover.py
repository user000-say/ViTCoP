"""BELA Dual COVER with *real* CLS scores and AnyRes-aware spatial coordinates.

Faithful to historical e013 REPLACE/COVER: anchors = CLS Top(K-B),
residual = complement of original CLS TopK, then B novelty replacements.
No norm-based proxy is used unless the caller explicitly requests an ablation.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

GRID = 24
CELLS = 4


def _validate_ids(local_patch_ids: Optional[torch.Tensor], n: int, device) -> torch.Tensor:
    ids = torch.arange(n, device=device, dtype=torch.long) if local_patch_ids is None else (
        torch.as_tensor(local_patch_ids, device=device, dtype=torch.long).flatten()
    )
    if ids.numel() != n or n > GRID * GRID or (n and (bool((ids < 0).any()) or bool((ids >= GRID*GRID).any()))):
        raise ValueError('local patch coordinates must be unique IDs in the original 24x24 crop')
    if ids.unique().numel() != n:
        raise ValueError('duplicate original crop-local patch IDs (unpad mapping corrupt)')
    return ids


def _relative_coverage(metric, saliency, anchors, residual, source_local_ids):
    """Historical 4x4 prototype formula, using original crop-local coordinates."""
    if residual.numel() == 0:
        return saliency[:0]
    cell_src = (source_local_ids // GRID // (GRID // CELLS)) * CELLS + (
        (source_local_ids % GRID) // (GRID // CELLS)
    )
    features = F.normalize(metric.float(), p=2, dim=-1, eps=1e-8)
    means = torch.zeros(CELLS * CELLS, features.shape[-1], device=metric.device, dtype=torch.float32)
    means.index_add_(0, cell_src.index_select(0, anchors), features.index_select(0, anchors))
    means = F.normalize(means, p=2, dim=-1, eps=1e-8)
    cos = (features.index_select(0, residual) * means.index_select(0, cell_src.index_select(0, residual))).sum(-1)
    novelty = 1 - cos.clamp(0, 1)
    return saliency.float().index_select(0, residual) * novelty


@torch.no_grad()
def dual_cover_one_crop(
    mid_nd: torch.Tensor,
    pen_nd: torch.Tensor,
    k: int,
    slots: int = 32,
    seed: int = 20261009,
    *,
    cls_scores: Optional[torch.Tensor] = None,
    local_patch_ids: Optional[torch.Tensor] = None,
    saliency_mode: str = 'cls_attention',
) -> torch.Tensor:
    """Select sorted row IDs of an AnyRes crop's *valid packed* patch table.

    cls_scores must be the ViT's real CLS->patch attention, already packed using
    the exact native AnyRes geometry. norm_proxy is ONLY a declared control.
    """
    if mid_nd.ndim != 2 or pen_nd.ndim != 2 or mid_nd.shape != pen_nd.shape:
        raise ValueError('mid and pen must have identical [N,D] shape')
    n = mid_nd.shape[0]
    if not (0 < k <= n):
        raise ValueError(f'invalid k={k} for crop capacity {n}')
    local_ids = _validate_ids(local_patch_ids, n, mid_nd.device)
    if saliency_mode == 'cls_attention':
        if cls_scores is None:
            raise RuntimeError('REAL_CLS_REQUIRED: missing CLS->patch attention')
        scores = torch.as_tensor(cls_scores, dtype=torch.float32, device=mid_nd.device)
    elif saliency_mode == 'norm_proxy':
        scores = pen_nd.float().norm(dim=-1)
    else:
        raise ValueError('saliency_mode must be cls_attention or norm_proxy')
    if scores.shape != (n,) or not bool(torch.isfinite(scores).all()):
        raise ValueError('CLS scores must be finite, aligned to native packed crop rows')
    b = min(int(slots), 32, k-1, n-k)
    if b < 0 or slots < 0:
        raise ValueError('slots must be nonnegative')
    base = scores.topk(k, largest=True, sorted=False).indices.sort().values
    if b == 0:
        return base
    anchors = scores.topk(k-b, largest=True, sorted=False).indices.sort().values
    mask = torch.ones(n, dtype=torch.bool, device=scores.device)
    mask[base] = False  # IMPORTANT: exclude original full TopK, not just Top(K-B)
    residual = torch.where(mask)[0]
    metric = torch.cat((mid_nd.float(), pen_nd.float()), dim=-1)
    utility = _relative_coverage(metric, scores, anchors, residual, local_ids)
    chosen = residual.index_select(0, utility.topk(b, largest=True, sorted=False).indices)
    picked = torch.cat((anchors, chosen)).sort().values
    if picked.numel() != k or picked.unique().numel() != k:
        raise RuntimeError('COVER produced invalid count or duplicate patch')
    return picked
