"""A tested callback bridge for the CURRENT host VLM's real pruning functions.

The host still owns full hidden/sequence/position/mask/KV compaction. This module
validates when the host selector/gather are called and guards against budgeting
and identity errors. The host must call begin_block() BEFORE layer forward and
end_block() AFTER forward, supplying actual visual counts from its real tensor.
"""
from __future__ import annotations
from typing import Any, Callable, Mapping
from .core import Schedule, TokenLedger, assert_real_trace


class LivePruneCoordinator:
    """One request, one schedule. Never reuse between independent requests.

    select: (post_block, target_k, patch_ids, host_state) -> sorted local indices
    gather: (host_state, local_indices) -> (new_host_state, actual_visual_count)
    Both callbacks must invoke the ORIGINAL R3 model's selection and physical
    gathering. The bridge never implements KV/position/mask operations.
    """
    def __init__(self, schedule: Schedule, entry_patch_ids,
                 select: Callable, gather: Callable):
        self.schedule = schedule
        self.ledger = TokenLedger(entry_patch_ids)
        if len(self.ledger.ids) != schedule.entry:
            raise ValueError('entry patch ID count differs from registered schedule')
        self.select, self.gather = select, gather
        self.completed = 0
        self.counts: list[int] = []
        self.events_log: list[dict] = []
        self._inside = False

    def begin_block(self, block_number: int, actual_visual_count: int) -> None:
        if self._inside or block_number != self.completed + 1 or block_number > self.schedule.n_layers:
            raise ValueError('invalid or repeated decoder block entry')
        expected = self.schedule.trace()[block_number - 1]
        if type(actual_visual_count) is not int or actual_visual_count != expected:
            raise ValueError(f'block {block_number} physical visual count {actual_visual_count} != {expected}')
        if actual_visual_count != len(self.ledger.ids):
            raise ValueError('host tensor and original patch ID ledger diverged')
        self.counts.append(actual_visual_count)
        self._inside = True

    def end_block(self, block_number: int, host_state: Any):
        if not self._inside or block_number != self.completed + 1:
            raise ValueError('finish called without matching start')
        target = self.schedule.expected_after(block_number)
        if target is not None:
            keep_local = tuple(self.select(block_number, target, self.ledger.ids, host_state))
            selected_ids = self.ledger.next_keep(keep_local, target)
            # Note: no ledger mutation until host physically succeeds.
            new_state, actual_after = self.gather(host_state, keep_local)
            if type(actual_after) is not int or actual_after != target:
                raise RuntimeError('HOST gather did not physically prune to target K')
            kept = self.ledger.commit(keep_local, target)
            assert kept == selected_ids
            self.events_log.append(dict(post_block=block_number,
                                        before=self.counts[-1], after=target,
                                        keep_local_indices=list(keep_local),
                                        original_patch_ids=list(kept)))
            host_state = new_state
        self.completed = block_number
        self._inside = False
        return host_state

    def finish_prefill(self) -> dict:
        if self.completed != self.schedule.n_layers or self._inside:
            raise ValueError('did not process entire decoder prefill')
        assert_real_trace(self.counts, self.schedule)
        return dict(schedule=self.schedule.name, w=sum(self.counts),
                    counts=list(self.counts), events=self.events_log,
                    final_original_patch_ids=list(self.ledger.ids))


class SameBudgetRouter:
    """Choose between prerecorded schedules with identical work and prefix.

    Use only feature observations available BEFORE the routing boundary.
    This does not learn a policy or infer future utility by itself.
    """
    def __init__(self, choices: Mapping[str, Schedule], decision_post_block: int):
        if not choices or type(decision_post_block) is not int or decision_post_block < 1:
            raise ValueError('invalid router configuration')
        self.choices = dict(choices)
        self.decision_post_block = decision_post_block
        entries = {s.entry for s in self.choices.values()}
        budgets = {s.budget for s in self.choices.values()}
        layers = {s.n_layers for s in self.choices.values()}
        if len(entries) != 1 or len(budgets) != 1 or len(layers) != 1:
            raise ValueError('router choices must share entry/W/depth')
        if decision_post_block >= next(iter(layers)):
            raise ValueError('routing boundary outside decoder')
        prefixes = {s.trace()[:decision_post_block] for s in self.choices.values()}
        if len(prefixes) != 1:
            raise ValueError('choices must have identical causal prefix before decision')

    def choose(self, name: str) -> Schedule:
        if name not in self.choices:
            raise ValueError('unregistered route')
        return self.choices[name]
