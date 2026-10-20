"""Bind rectangular recycling to existing source-bound selectors, batch one."""
import torch
if __package__:
    from .current_torch import recycle_current
else:
    from current_torch import recycle_current


class CurrentState:
    def __init__(self, patch_ids, coords, groups, positions, events=('first',),
                 alpha=.5, cosine_floor=.8, radius=2., prefix_only=False,
                 debug=False):
        n = len(patch_ids)
        if patch_ids.ndim != 1 or patch_ids.dtype != torch.long:
            raise ValueError('unique on-device LongTensor patch IDs required')
        for value in (groups, positions):
            if value.shape != (n,) or value.device != patch_ids.device or value.dtype != torch.long:
                raise ValueError('aligned on-device groups/positions required')
        if coords.shape != (n, 2) or coords.device != patch_ids.device:
            raise ValueError('aligned coordinates required')
        if not set(events) <= {'first', 'final'}:
            raise ValueError('invalid events')
        if debug and patch_ids.unique().numel() != n:
            raise ValueError('duplicate original IDs')
        self.ids, self.coords, self.groups, self.positions = patch_ids, coords, groups, positions
        self.events, self.seen = set(events), set()
        self.kwargs = dict(alpha=alpha, cosine_floor=cosine_floor, radius=radius,
                           prefix_only=prefix_only, debug=debug)

    def compress(self, event, active_hidden, keep, current_metric=None):
        if event not in ('first', 'final') or event in self.seen or (event == 'final' and 'first' not in self.seen):
            raise ValueError('invalid event order')
        if len(active_hidden) != len(self.ids):
            raise ValueError('active identity diverged')
        kwargs = dict(self.kwargs)
        if event not in self.events:
            kwargs['alpha'] = 0.
        out, audit = recycle_current(active_hidden, keep, metric=current_metric,
            coords=self.coords, groups=self.groups, positions=self.positions, **kwargs)
        self.ids, self.coords = self.ids[keep], self.coords[keep]
        self.groups, self.positions = self.groups[keep], self.positions[keep]
        self.seen.add(event)
        return out, self.ids, audit
