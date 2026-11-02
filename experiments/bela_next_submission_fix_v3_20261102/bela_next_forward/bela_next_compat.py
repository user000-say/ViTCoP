"""BELA historical operators applied to *native packed patch* namespace.

No external absolute-path imports. No saliency proxy in the default arm.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from .dual_cover import dual_cover_one_crop
from bela_next_budget.crops import crop_quota_from_ids
from bela_query_anchor.llava_adapter import apply_llava_query_anchor
from bela_query_anchor.query_anchor import AnchorConfig


def _assign_slots(quotas, capacities, slots):
    """Largest-remainder allocation, respecting each group's actual residual capacity."""
    caps = [max(0, min(32, q-1, capacity-q)) for q, capacity in zip(quotas, capacities)]
    usable = min(int(slots), sum(caps))
    if usable <= 0:
        return [0]*len(quotas), 0
    w = [float(q) if cap else 0.0 for q, cap in zip(quotas, caps)]
    alloc = [0]*len(quotas)
    for _ in range(usable):
        valid = [i for i in range(len(quotas)) if alloc[i] < caps[i]]
        chosen = max(valid, key=lambda i: (w[i]/(alloc[i]+1), -i))
        alloc[chosen] += 1
    return alloc, usable


@torch.no_grad()
def original_dual(*, mid: torch.Tensor, pen: torch.Tensor, keep: int, slots: int,
                  seed: int, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    if context is None:
        context = {}
    layout = context.get('native_layout')
    saliency = context.get('packed_cls_scores')
    mode = context.get('saliency_mode', 'cls_attention')
    if layout is None:
        raise RuntimeError('NATIVE_LAYOUT_REQUIRED: crop provenance is mandatory on NeXT')
    if mode == 'cls_attention' and saliency is None:
        raise RuntimeError('REAL_CLS_REQUIRED: actual penultimate CLS attention unavailable')
    crop = layout.origin_crop[layout.patch_mask].to(device=mid.device, dtype=torch.long)
    local_ids = layout.origin_local_patch[layout.patch_mask].to(device=mid.device, dtype=torch.long)
    if crop.numel() != mid.shape[0] or pen.shape[0] != mid.shape[0]:
        raise RuntimeError('crop provenance/feature length mismatch')
    if saliency is not None:
        saliency = saliency.to(device=mid.device, dtype=torch.float32)
        if saliency.shape != (mid.shape[0],):
            raise RuntimeError('CLS saliency does not match packed patch namespace')
    uniq = list(dict.fromkeys(int(c) for c in crop.tolist()))
    if not uniq or any(c < 0 for c in uniq):
        raise RuntimeError('invalid crop IDs')
    capacities = [int((crop == c).sum()) for c in uniq]
    policy = context.get('crop_policy', 'uniform')
    if policy not in ('uniform', 'thumbnail_boost'):
        raise RuntimeError(f'Unknown crop policy: {policy}')
    quotas = crop_quota_from_ids(capacities, int(keep), policy=policy,
                                 thumbnail_crop=0, global_weight=float(context.get('thumbnail_weight', 1.5)),
                                 min_per_crop=1)
    alloc, usable = _assign_slots(quotas, capacities, int(slots))
    parts = []
    for c, qq, b in zip(uniq, quotas, alloc):
        rows = torch.where(crop == c)[0]
        picked = dual_cover_one_crop(
            mid.index_select(0, rows), pen.index_select(0, rows), int(qq),
            slots=b, seed=int(seed)+int(c),
            cls_scores=None if saliency is None else saliency.index_select(0, rows),
            local_patch_ids=local_ids.index_select(0, rows), saliency_mode=mode,
        )
        parts.append(rows.index_select(0, picked))
    result = torch.cat(parts).sort().values
    if result.numel() != int(keep) or result.unique().numel() != int(keep):
        raise RuntimeError('Dual COVER invalid global selection count')
    context['dual_slots_per_crop'] = alloc
    context['dual_quotas'] = quotas
    context['dual_crop_ids'] = uniq
    context['dual_slots_global'] = int(slots)
    context['dual_slots_effective'] = usable
    context['dual_saliency_mode'] = mode
    return result


@torch.no_grad()
def original_qa(*, mid: torch.Tensor, pen: torch.Tensor, question: torch.Tensor,
                question_ids: Optional[torch.Tensor], initial_keep: torch.Tensor,
                max_swaps: int, donor: int, drop: int, seed: int,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    if context is None:
        context = {}
    model = context.get('model')
    if model is None or question_ids is None or question_ids.numel() == 0:
        raise RuntimeError('QA requires model and validated original question IDs')
    projected = context.get('projected_full_vision')
    if projected is None or projected.shape[0] != pen.shape[0]:
        raise RuntimeError('QA requires packed original projected visual features [N,D]')
    metric = torch.cat((mid.float(), pen.float()), dim=-1)
    result = apply_llava_query_anchor(
        mode='s1_anchor', native_keep_ids=initial_keep.long(), dual_metric=metric,
        full_vision_features_for_projector=pen,
        projected_full_vision=projected,
        question_only_ids=question_ids.long().reshape(-1),
        mm_projector=model.multi_modal_projector,
        llm_input_embeddings=model.get_input_embeddings(),
        config=AnchorConfig(max_swaps=int(max_swaps), donor_pool=int(donor), drop_pool=int(drop)),
        random_seed=int(seed), diagnostics=True,
    )
    ids = result.keep_ids.long().to(initial_keep.device).sort().values
    if ids.numel() != initial_keep.numel() or ids.unique().numel() != ids.numel():
        raise RuntimeError('QA changed K or generated duplicate IDs')
    if int(result.swaps) != int((~torch.isin(ids, initial_keep)).sum()):
        raise RuntimeError('QA swap count disagrees with final kept set')
    context['qa_reason'] = result.reason
    context['qa_swaps_reported'] = int(result.swaps)
    context['qa_records_n'] = len(result.records or [])
    # QA deliberately remains global: measure crop drift rather than silently altering its algorithm.
    crop = context['native_layout'].origin_crop[context['native_layout'].patch_mask]
    context['qa_crop_counts'] = {str(int(c)): int((crop[ids] == c).sum()) for c in crop.unique()}
    return ids
