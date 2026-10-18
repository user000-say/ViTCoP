"""State tracking around the existing selector and the existing model gather.

This module never rewrites positions, masks, or past KV. The VLM integration
uses the original compaction and replaces only selected visual hidden vectors.
"""
import hashlib
import numpy as np
if __package__:
    from .recycle import AffinityCache, recycle
else:
    from recycle import AffinityCache, recycle


def input_seed(sample_id, seed):
    payload = f"{sample_id}\x00{int(seed)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class RecyclingState:
    def __init__(self, source_id, entry_features, entry_coords, entry_patch_ids,
                 operator_config, groups=None, seed=0):
        self.source_id = str(source_id)
        self.config = dict(operator_config)
        self.seed = input_seed(source_id, seed)
        self.patch_ids = np.asarray(entry_patch_ids)
        if self.patch_ids.ndim != 1 or len(np.unique(self.patch_ids)) != len(self.patch_ids):
            raise ValueError("unique original patch IDs required")
        self.slots = np.arange(len(self.patch_ids))
        self.cache = None
        self.seen = set()
        if self.config.get("events"):
            if len(entry_features) != len(self.patch_ids):
                raise ValueError("entry features/IDs disagree")
            self.cache = AffinityCache(source_id, entry_features, entry_coords, groups)

    def compress(self, event, active_hidden, keep, existing_question_mass=None):
        if event not in ("first", "final") or event in self.seen or (event == "final" and "first" not in self.seen):
            raise ValueError("invalid or duplicate event")
        if len(active_hidden) != len(self.slots):
            raise ValueError("active hidden states and patch identity diverged")
        enabled = event in self.config.get("events", [])
        kwargs = self.cache.view(self.source_id, self.slots) if enabled else {}
        if enabled and self.config["weights"] == "existing_question_mass":
            if existing_question_mass is None:
                raise ValueError("query weights require a real existing attention reader")
            kwargs["weights"] = existing_question_mass
        result = recycle(active_hidden, keep, **kwargs,
            alpha=self.config["alpha"] if enabled else 0.,
            cosine_floor=self.config.get("cosine_floor", .8),
            radius=self.config.get("radius", 2.),
            assignment=self.config.get("assignment", "semantic"),
            preserve_norm=self.config.get("preserve_norm", False), seed=self.seed)
        self.slots = self.slots[result.keep]
        self.seen.add(event)
        return result, self.patch_ids[self.slots]
