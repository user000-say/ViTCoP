"""Thin callback into the existing E013 REPLACE/COVER. No new coverage formula."""
from llava.model.multimodal_encoder.e013_stage1_support import e013_choose


def legacy_callback(*, metric_features, scores, patch_ids, k, context=None):
    if int(k) != 172:
        raise RuntimeError("BLOCKED_CAPACITY: historical COVER is K=172 only")
    slots = int((context or {}).get("replacement_slots", 32))
    selected, _extra, _original = e013_choose(
        metric_features, scores, "REPLACE", "COVER", slots, 20261009, 0)
    # e013 returns original patch coordinates. Map them to active-row indices.
    index = {int(pid): i for i, pid in enumerate(patch_ids.detach().cpu().tolist())}
    local = [index[int(x)] for x in selected.detach().cpu().tolist()]
    return selected.new_tensor(local, dtype=selected.dtype)
