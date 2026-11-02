"""Original 1.5-7B Dual COVER + QA-COVER wrappers for native NeXT packed space.

Selection IDs are indices into the *packed patch-only* stream (newlines excluded),
matching BELA_NEXT_NATIVE_FIX_KIT's run_original_bela contract.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from bela_next_forward.dual_cover import dual_cover_one_crop


def original_dual(
    *,
    mid: torch.Tensor,
    pen: torch.Tensor,
    keep: int,
    slots: int,
    seed: int,
    context: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Crop-grouped Dual COVER on native packed patch features."""
    context = context or {}
    layout = context.get("native_layout")
    if layout is None:
        # Fallback: treat whole packed stream as one group
        return dual_cover_one_crop(mid, pen, int(keep), slots=int(slots), seed=int(seed))

    crop = layout.origin_crop[layout.patch_mask].to(device=mid.device, dtype=torch.long)
    if int(crop.numel()) != int(mid.shape[0]):
        raise RuntimeError(
            f"native_layout patch crop map {crop.numel()} != mid N {mid.shape[0]}"
        )
    # Preserve first-seen crop order (base then tiles)
    uniq = []
    for c in crop.tolist():
        if c not in uniq:
            uniq.append(int(c))
    capacities = [int((crop == c).sum().item()) for c in uniq]
    from bela_next_budget.crops import crop_quota_from_ids

    quotas = crop_quota_from_ids(capacities, int(keep), policy="uniform", min_per_crop=1)
    # CRITICAL: slots is a GLOBAL Dual COVER budget (32 on 1.5-7B / one 576 grid).
    # Re-applying slots=32 on EVERY NeXT crop inflates COVER ~Nx (e.g. 5×160/320),
    # which destroys saliency anchors and collapses ACC despite higher resolution.
    # Distribute the same global slot budget proportionally across crop quotas.
    keep_i = int(keep)
    slots_i = int(slots)
    qq_list = []
    for c, q in zip(uniq, quotas):
        rows_n = int((crop == c).sum().item())
        qq_list.append(min(int(q), rows_n))
    qq_sum = sum(qq_list) or 1
    raw = [slots_i * qq / qq_sum for qq in qq_list]
    slots_per = [int(x) for x in raw]
    rem = slots_i - sum(slots_per)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - slots_per[i], -qq_list[i], i), reverse=True)
    for i in order:
        if rem <= 0:
            break
        slots_per[i] += 1
        rem -= 1
    parts = []
    for c, qq, slots_c in zip(uniq, qq_list, slots_per):
        rows = torch.nonzero(crop == c, as_tuple=True)[0]
        if qq <= 0:
            continue
        # Cap per-crop COVER so anchors still dominate (mirror 1.5 ratio).
        slots_c = max(0, min(int(slots_c), int(qq) - 1 if qq > 1 else 0, 32))
        local = dual_cover_one_crop(
            mid.index_select(0, rows),
            pen.index_select(0, rows),
            qq,
            slots=slots_c,
            seed=int(seed) + int(c),
        )
        parts.append(rows.index_select(0, local))
    if isinstance(context, dict):
        context["dual_slots_per_crop"] = list(slots_per)
        context["dual_quotas"] = list(qq_list)
        context["dual_slots_global"] = slots_i
    out = torch.cat(parts, dim=0).sort().values
    if int(out.numel()) != int(keep):
        # capacities may have forced fewer than keep when unpad is harsh
        if int(out.numel()) < int(keep):
            raise RuntimeError(f"Dual COVER emitted {out.numel()} < keep={keep}")
        out = out[:keep]
    return out


def original_qa(
    *,
    mid: torch.Tensor,
    pen: torch.Tensor,
    question: torch.Tensor,
    question_ids: Optional[torch.Tensor],
    initial_keep: torch.Tensor,
    max_swaps: int,
    donor: int,
    drop: int,
    seed: int,
    context: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Global B8 QA-COVER via historical bela_query_anchor donor-swap."""
    context = context or {}
    model = context.get("model")
    if model is None:
        raise RuntimeError("QA requires context['model']")
    if question_ids is None or int(question_ids.numel()) == 0:
        raise RuntimeError("QA requires real question_ids")

    import sys
    from pathlib import Path

    root = str(Path("/home/xiangyue/ViTCoP/experiments"))
    if root not in sys.path:
        sys.path.insert(0, root)
    from bela_query_anchor.llava_adapter import apply_llava_query_anchor
    from bela_query_anchor.query_anchor import AnchorConfig

    # Dual metric = concat(mid, pen) as in e013 REPLACE/COVER
    metric = torch.cat([mid.float(), pen.float()], dim=-1)
    mm_proj = model.multi_modal_projector
    embed = model.get_input_embeddings()
    cfg = AnchorConfig(
        max_swaps=int(max_swaps),
        donor_pool=int(donor),
        drop_pool=int(drop),
    )
    result = apply_llava_query_anchor(
        mode="s1_anchor",
        native_keep_ids=initial_keep.long(),
        dual_metric=metric,
        full_vision_features_for_projector=pen,
        question_only_ids=question_ids.long().reshape(-1),
        mm_projector=mm_proj,
        llm_input_embeddings=embed,
        config=cfg,
        random_seed=int(seed),
        diagnostics=True,
    )
    # AnchorResult: .keep or .ids — support both
    keep = getattr(result, "keep_ids", None)
    if keep is None:
        keep = getattr(result, "keep", None)
    if keep is None:
        keep = getattr(result, "selected_ids", None)
    if keep is None:
        raise RuntimeError(f"QA result missing keep ids: {type(result)} {dir(result)}")
    keep = keep.long().to(device=initial_keep.device).sort().values
    if int(keep.numel()) != int(initial_keep.numel()):
        raise RuntimeError(f"QA changed K: {keep.numel()} vs {initial_keep.numel()}")
    # Surface QA reason for smoke/ACC traces (was silent no_safe_query_anchor).
    if isinstance(context, dict):
        context["qa_reason"] = getattr(result, "reason", None)
        context["qa_swaps_reported"] = int(getattr(result, "swaps", 0) or 0)
        context["qa_records_n"] = (
            len(result.records) if getattr(result, "records", None) is not None else 0
        )
    return keep
