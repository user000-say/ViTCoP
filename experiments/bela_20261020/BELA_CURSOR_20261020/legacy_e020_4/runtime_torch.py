"""Per-input GPU state. The original model still owns position/mask/KV gather."""
import torch
if __package__:
    from .runtime import input_seed
    from .torch_backend import cosine_cache, recycle
else:
    from runtime import input_seed
    from torch_backend import cosine_cache, recycle


class RecyclingState:
    def __init__(self, source_id, entry_features, entry_coords, entry_patch_ids,
                 operator_config, groups=None, seed=0, debug=False):
        if entry_patch_ids.ndim != 1 or entry_patch_ids.dtype != torch.long:
            raise ValueError("original patch IDs must be a LongTensor")
        self.source_id, self.config = str(source_id), dict(operator_config)
        self.seed, self.debug = input_seed(source_id, seed), debug
        self.patch_ids = entry_patch_ids
        self.slots = torch.arange(len(entry_patch_ids), device=entry_patch_ids.device)
        self.seen = set()
        self.affinity = self.coords = self.groups = None
        if self.config.get("events"):
            if entry_features is None or entry_coords is None or len(entry_features) != len(entry_patch_ids):
                raise ValueError("entry features/coords/IDs must agree")
            if entry_features.device != entry_patch_ids.device or entry_coords.device != entry_patch_ids.device:
                raise ValueError("entry state must remain on model device")
            if entry_coords.shape != (len(entry_patch_ids), 2):
                raise ValueError("original patch coordinates required")
            if groups is not None and (groups.shape != entry_patch_ids.shape or groups.device != entry_patch_ids.device):
                raise ValueError("invalid crop groups")
            self.affinity, self.coords, self.groups = cosine_cache(entry_features), entry_coords, groups
        if debug and entry_patch_ids.unique().numel() != entry_patch_ids.numel():
            raise ValueError("duplicate original patch ID")

    def compress(self, event, active_hidden, keep, existing_question_mass=None):
        if event not in ("first", "final") or event in self.seen or (event == "final" and "first" not in self.seen):
            raise ValueError("invalid/duplicate pruning event")
        if len(active_hidden) != len(self.slots) or active_hidden.device != self.slots.device:
            raise ValueError("hidden states and active patch identity disagree")
        enabled = event in self.config.get("events", [])
        kwargs = {}
        if enabled:
            kwargs = dict(affinity=self.affinity[self.slots][:, self.slots],
                          coords=self.coords[self.slots],
                          groups=None if self.groups is None else self.groups[self.slots])
            if self.config["weights"] == "existing_question_mass":
                if existing_question_mass is None:
                    raise ValueError("real existing question weights required")
                kwargs["weights"] = existing_question_mass
        out, audit = recycle(active_hidden, keep, **kwargs,
            alpha=self.config["alpha"] if enabled else 0.,
            cosine_floor=self.config.get("cosine_floor", .8), radius=self.config.get("radius", 2.),
            assignment=self.config.get("assignment", "semantic"),
            preserve_norm=self.config.get("preserve_norm", False), seed=self.seed, debug=self.debug)
        self.slots = self.slots[keep]
        self.seen.add(event)
        return out, self.patch_ids[self.slots], audit
