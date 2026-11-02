"""Historical Dual COVER (e013 REPLACE/COVER) applied crop-wise for NeXT grids.

NeXT A0–A4 per-crop quotas (e.g. 64/58/52) are outside the LLaVA-1.5 e013
core_k whitelist; formula is identical — only the whitelist is lifted here.

When a crop has an unpad-valid pool smaller than 576, COVER runs inside that
pool and returns original local patch IDs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

_E013 = None


def _e013():
    global _E013
    if _E013 is None:
        p = Path("/home/xiangyue/ViTCoP/LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py")
        spec = importlib.util.spec_from_file_location("e013_stage1_support", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _E013 = mod
    return _E013


@torch.no_grad()
def dual_cover_one_crop(
    mid_nd: torch.Tensor,
    pen_nd: torch.Tensor,
    k: int,
    slots: int = 32,
    seed: int = 20261009,
) -> torch.Tensor:
    """Return sorted *row* IDs in [0,N) for one feature table (REPLACE/COVER)."""
    if mid_nd.ndim != 2 or pen_nd.ndim != 2:
        raise ValueError("mid/pen must be [N,D]")
    if mid_nd.shape[0] != pen_nd.shape[0]:
        raise ValueError("mid/pen N mismatch")
    n = int(mid_nd.shape[0])
    if not (0 < k <= n):
        raise ValueError(f"invalid k={k} for N={n}")
    e = _e013()
    scores = pen_nd.float().norm(dim=-1)
    metric = torch.cat([mid_nd.float(), pen_nd.float()], dim=-1)
    b = max(0, min(int(slots), 32, k - 1 if k > 1 else 0))
    try:
        if n == 576:
            sel, _extra, _orig = e.e013_choose(
                metric, scores, "REPLACE", "COVER", b, seed, 0, core_k=k
            )
            ids = sel.long().sort().values
        else:
            raise ValueError("non-576 uses lifted path")
    except ValueError:
        total_k = int(k)
        base = scores.topk(total_k, largest=True, sorted=False).indices.sort().values
        if b == 0:
            ids = base
        else:
            base_mask = torch.ones(n, dtype=torch.bool, device=metric.device)
            base_mask[base] = False
            residual = torch.where(base_mask)[0]
            anchors_k = total_k - b
            anchors = scores.topk(anchors_k, largest=True, sorted=False).indices.sort().values
            utility = e._core_relative_coverage(metric, scores, anchors, residual)
            local = utility.topk(b, largest=True, sorted=False).indices
            chosen = residual.index_select(0, local)
            ids = torch.cat((anchors, chosen), dim=0).sort().values
    if int(ids.unique().numel()) != k:
        raise RuntimeError(f"COVER unique={ids.unique().numel()} != k={k}")
    if int(ids.min()) < 0 or int(ids.max()) >= n:
        raise RuntimeError("COVER id out of range")
    return ids


@torch.no_grad()
def dual_cover_one_crop_pool(
    mid_576: torch.Tensor,
    pen_576: torch.Tensor,
    k: int,
    pool_local_ids: Optional[torch.Tensor] = None,
    slots: int = 32,
    seed: int = 20261009,
) -> torch.Tensor:
    """COVER on full 576 or a valid local-id pool; returns sorted original local ids."""
    if mid_576.ndim != 2 or mid_576.shape[0] != 576:
        raise ValueError("mid must be [576,D]")
    if pool_local_ids is None:
        return dual_cover_one_crop(mid_576, pen_576, k, slots=slots, seed=seed)
    pool = pool_local_ids.to(device=mid_576.device, dtype=torch.long).unique().sort().values
    if pool.numel() == 0:
        raise RuntimeError("empty valid pool")
    kk = min(int(k), int(pool.numel()))
    if kk <= 0:
        raise RuntimeError("k<=0 after pool clamp")
    mid_p = mid_576.index_select(0, pool)
    pen_p = pen_576.index_select(0, pool)
    row = dual_cover_one_crop(mid_p, pen_p, kk, slots=slots, seed=seed)
    return pool.index_select(0, row).sort().values


@torch.no_grad()
def dual_cover_five_crops(
    mid_b576d: torch.Tensor,
    pen_b576d: torch.Tensor,
    quotas: Sequence[int],
    slots: int = 32,
    valid_pools: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """mid/pen: [C,576,D] → flat keep ids in global crop*576+local and per-crop ids."""
    n_crops = int(mid_b576d.shape[0])
    if pen_b576d.shape[0] != n_crops:
        raise ValueError("mid/pen crop count mismatch")
    if len(quotas) != n_crops:
        raise ValueError("need quotas for each crop")
    if valid_pools is not None and len(valid_pools) != n_crops:
        raise ValueError("valid_pools length mismatch")
    per = []
    flat = []
    for ci, k in enumerate(quotas):
        pool = None if valid_pools is None else valid_pools[ci]
        local = dual_cover_one_crop_pool(
            mid_b576d[ci], pen_b576d[ci], int(k), pool_local_ids=pool, slots=slots
        )
        per.append(local)
        flat.append(local + ci * 576)
    return torch.cat(flat, dim=0).sort().values, per
