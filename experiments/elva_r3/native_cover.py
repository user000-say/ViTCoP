"""Thin callback into the existing E013 REPLACE/COVER. No new coverage formula."""
from llava.model.multimodal_encoder.e013_stage1_support import e013_choose

# Verified entry budgets for the same REPLACE/COVER gain (slots<=32).
_ALLOWED_K = {128, 160, 172}


def legacy_callback(*, metric_features, scores, patch_ids, k, context=None):
    k = int(k)
    if k not in _ALLOWED_K:
        raise RuntimeError(
            f"BLOCKED_CAPACITY: historical COVER allows K in {_ALLOWED_K}, got {k}"
        )
    slots = int((context or {}).get("replacement_slots", 32))
    selected, _extra, _original = e013_choose(
        metric_features, scores, "REPLACE", "COVER", slots, 20261009, 0, core_k=k
    )
    # e013 returns original patch coordinates. Map them to active-row indices.
    index = {int(pid): i for i, pid in enumerate(patch_ids.detach().cpu().tolist())}
    local = [index[int(x)] for x in selected.detach().cpu().tolist()]
    return selected.new_tensor(local, dtype=selected.dtype)
