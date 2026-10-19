"""Thin source-bound runtime contract, independent of a tensor framework.

Native callbacks operate ON DEVICE. This file never converts CUDA tensors
to NumPy. Adapters must compact position_ids/masks and maintain per-layer
cache_position/cache support with the actual model's native semantics.
"""
from dataclasses import dataclass
from .budget import Trace


@dataclass
class BoundRuntime:
    schedule: object
    select_native: object
    compact_native: object
    inspect_native: object

    def __post_init__(self):
        self.trace = Trace(self.schedule.work, self.schedule.depth)
        self.events = dict(self.schedule.events)
        self.audit = []

    def before_block(self, block, state):
        view = self.inspect_native(state)
        self.trace.before_block(block, len(view["active_ids"]))
        expected = self.schedule.trace[block-1]
        if len(view["active_ids"]) != expected:
            raise ValueError("wrong hook location or keep count")

    def after_block(self, completed, state, *, selector, context=None):
        if completed != len(self.trace.counts):
            raise ValueError("pruning must follow the recorded block")
        if completed not in self.events:
            return state
        before = self.inspect_native(state)
        keep = self.select_native(state, k=self.events[completed], selector=selector, context=context)
        # keep uses ORIGINAL globally unique patch IDs, not sequence columns.
        ids = tuple(map(int, keep))
        if len(ids) != self.events[completed] or len(set(ids)) != len(ids) or not set(ids).issubset(before["active_ids"]):
            raise ValueError("selector returned duplicate/unknown IDs or wrong K")
        result = self.compact_native(state, keep_ids=ids)
        after = self.inspect_native(result)
        if (set(after["active_ids"]) != set(ids)
                or before["protected_positions"] != after["protected_positions"]
                or before["backend"] != after["backend"]):
            raise ValueError("compaction changed protected positions/backend/identity")
        if "visual_positions" in before:
            expected_positions = {i: before["visual_positions"][i] for i in ids}
            if after.get("visual_positions") != expected_positions:
                raise ValueError("compaction changed original visual RoPE positions")
        self.audit.append(dict(completed=completed, selector=selector,
                               before=before["active_ids"], after=after["active_ids"]))
        return result

    def finish(self):
        actual = self.trace.finish()
        self.schedule.validate_trace(actual)
        if len(self.audit) != len(self.events):
            raise ValueError("missing or duplicate pruning hook")
        return actual
