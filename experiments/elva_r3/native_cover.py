"""Thin callback into the existing E013 REPLACE/COVER. No new coverage formula.

CVPR2027 handoff Stage1: production COVER accepts budget=K0 via ``core_k``.
Allowed K0 match handoff profiles + G1 F3:
  96 (EASY), 128 (G1 F3), 160 (NORMAL / G1 F2), 172 (HARD / frozen anchor).
Replacement slots stay 32 (same REPLACE/COVER criterion as historical K172).
Not PCOV / not pool172_then_topk.
"""
from llava.model.multimodal_encoder.e013_stage1_support import e013_choose

# Handoff PolicyProfile k0 + G1 F3 entry128.
_ALLOWED_K = {96, 128, 160, 172}
_COVER_SLOTS = 32


def legacy_callback(*, metric_features, scores, patch_ids, k, context=None):
    k = int(k)
    if k not in _ALLOWED_K:
        raise RuntimeError(
            f"BLOCKED_CAPACITY: COVER@K0 allows K in {_ALLOWED_K}, got {k}"
        )
    slots = int((context or {}).get("replacement_slots", _COVER_SLOTS))
    if slots != _COVER_SLOTS:
        raise RuntimeError(
            f"BLOCKED_SOURCE: COVER@K0 keeps historical REPLACE slots={_COVER_SLOTS}, got {slots}"
        )
    selected, _extra, _original = e013_choose(
        metric_features, scores, "REPLACE", "COVER", slots, 20261009, 0, core_k=k
    )
    # e013 returns original patch coordinates. Map them to active-row indices.
    index = {int(pid): i for i, pid in enumerate(patch_ids.detach().cpu().tolist())}
    local = [index[int(x)] for x in selected.detach().cpu().tolist()]
    return selected.new_tensor(local, dtype=selected.dtype)
