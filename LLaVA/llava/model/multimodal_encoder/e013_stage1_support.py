"""E013-P1 (2026-10-09): frozen, label-free Stage1 headroom selectors.

Research-use candidates, NOT a validated adaptive algorithm.  This module is
imported only when E013_ENABLE=True.  Selection works on *CLIP output* patches
before mm_projector, so no second CLIP forward or extra text encoder is added.

Definitions:
  Persistent Core: 172 patch IDs, always QED-eligible.
  EXTRA: CLS Top172 + b transient positions; transients are NOT QED-eligible.
  REPLACE: CLS Top(172-b) + b residual patches, same 172 total positions.
  Residual for either arm excludes the ORIGINAL CLS Top172.  Every control
  receives identical residual candidates and the same budget b.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

CORE_K = 172
GRID = 24
CELLS = 4


def _core_relative_coverage(tokens, scores, anchor_ids, residual_ids):
    """Cheap spatial-prototype novelty, not globally optimal DPP/k-center.

    A core-aware 4x4 prototype m_g is the mean of normalized selected CLIP
    features in cell g, then re-normalized.  For a residual i in cell g:
      score(i) = CLS_attention(i) * [1 - max(0, cos(x_i, m_g))].
    An empty cell has m_g=0 and thus maximal novelty multiplier 1.
    No learned coefficients, N^2 token similarity or Python token loops.
    """
    device = tokens.device
    assert tokens.ndim == 2
    row = torch.arange(GRID * GRID, device=device) // GRID
    col = torch.arange(GRID * GRID, device=device) % GRID
    cell = (row // (GRID // CELLS)) * CELLS + col // (GRID // CELLS)
    features = F.normalize(tokens.float(), p=2, dim=-1, eps=1e-8)
    anchor_cell = cell.index_select(0, anchor_ids)
    means = torch.zeros(CELLS * CELLS, features.shape[-1],
                        dtype=torch.float32, device=device)
    means.index_add_(0, anchor_cell, features.index_select(0, anchor_ids))
    means = F.normalize(means, p=2, dim=-1, eps=1e-8)
    resid_cell = cell.index_select(0, residual_ids)
    cosine = (features.index_select(0, residual_ids) *
              means.index_select(0, resid_cell)).sum(dim=-1)
    novelty = 1.0 - cosine.clamp(min=0.0, max=1.0)
    return scores.float().index_select(0, residual_ids) * novelty


def e013_choose(tokens, cls_scores, variant, pick, budget,
                seed=20261009, request_index=0):
    """Return (persistent_patch_ids, appended_patch_ids, original_patch_ids).

    ALL IDs refer to original 576 CLIP patches, are unique and spatially sorted.
    NULL's appended IDs are empty: caller must allocate b zero CLIP vectors.
    DUPLICATE appends b already-present core tokens as an unchanged-content
    control (these slots still add attention / normalization and RoPE).
    Random uses a seeded PyTorch generator, independent of the global RNG,
    keyed by request_index: RUNNER MUST keep the exact same sample order.
    """
    if tokens.ndim != 2 or tokens.shape[0] != GRID*GRID:
        raise ValueError('[E013] requires [576, dim] CLIP patch tokens')
    if cls_scores.ndim != 1 or cls_scores.shape[0] != GRID*GRID:
        raise ValueError('[E013] requires 576 raw CLS scores')
    # E013-P1: no GPU->CPU tensor checks in the HOT selection path.
    # Nonfinite checks run only when E013_AUDIT=True in clip_encoder.
    variant = str(variant).upper()
    pick = str(pick).upper()
    b = int(budget)
    if variant not in {'EXTRA', 'REPLACE'}:
        raise ValueError('[E013] variant must be EXTRA or REPLACE')
    if pick not in {'ATTN', 'RANDOM', 'COVER', 'NULL', 'DUPLICATE'}:
        raise ValueError('[E013] pick must be ATTN/RANDOM/COVER/NULL/DUPLICATE')
    if pick in {'NULL', 'DUPLICATE'} and variant != 'EXTRA':
        raise ValueError('[E013] NULL/DUPLICATE are only valid for EXTRA')
    if b < 0 or b > 32:
        raise ValueError('[E013] budget must be 0..32')

    base = cls_scores.topk(CORE_K, largest=True, sorted=False).indices
    base = base.sort().values
    if b == 0:
        return base, base[:0], base
    base_mask = torch.ones(GRID*GRID, dtype=torch.bool, device=tokens.device)
    base_mask[base] = False
    residual = torch.where(base_mask)[0]
    anchors_k = CORE_K if variant == 'EXTRA' else CORE_K-b
    anchors = cls_scores.topk(anchors_k, largest=True, sorted=False).indices.sort().values

    if pick == 'NULL':
        chosen = residual[:0]
    elif pick == 'DUPLICATE':
        chosen = base.index_select(0, cls_scores.index_select(0, base).topk(b).indices).sort().values
    elif pick == 'ATTN':
        local = cls_scores.index_select(0, residual).topk(b, largest=True, sorted=False).indices
        chosen = residual.index_select(0, local).sort().values
    elif pick == 'RANDOM':
        # E013-P1: random accuracy negative control; benchmark its RNG costs
        # separately, not as an inference-optimized selector.
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(int(seed) + 1000003 * int(request_index))
        random_scores = torch.rand((residual.numel(),), device=tokens.device, generator=generator)
        local = random_scores.topk(b, largest=True, sorted=False).indices
        chosen = residual.index_select(0, local).sort().values
    else:
        utility = _core_relative_coverage(tokens, cls_scores, anchors, residual)
        local = utility.topk(b, largest=True, sorted=False).indices
        chosen = residual.index_select(0, local).sort().values

    if variant == 'REPLACE':
        persistent = torch.cat((anchors, chosen), dim=0).sort().values
        transient = chosen[:0]
    else:
        persistent = base
        transient = chosen

    # Sets are disjoint by construction (original Core vs its complement).
    # Avoid torch.unique and .item() synchronizations during inference.
    if persistent.numel() != CORE_K:
        raise RuntimeError('[E013] wrong persistent length')
    return persistent, transient, base
