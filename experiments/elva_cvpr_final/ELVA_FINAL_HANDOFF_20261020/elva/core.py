"""Runtime contracts for ELVA VLM pruning experiments (2026-10-20).

The unmodified host VLM owns RoPE/position IDs, causal mask, KV cache,
text/vision sequence assembly and all physical compaction. This module NEVER
claims to implement those model-specific operations. All layer counts refer
to the number of *completed* decoder blocks, starting with zero before block 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import math


@dataclass(frozen=True)
class Schedule:
    """Physical visual token count for each decoder block.

    events contains (completed_block_count, count_for_next_block).
    E.g. entry=172, events=((4,90),(12,32)): 4/8/20 blocks.
    """
    name: str
    entry: int
    events: tuple[tuple[int, int], ...]
    n_layers: int = 32
    full_visual_tokens: int = 576

    def __post_init__(self):
        if type(self.entry) is not int or not (1 <= self.entry <= self.full_visual_tokens):
            raise ValueError('invalid entry count')
        if type(self.n_layers) is not int or self.n_layers < 1:
            raise ValueError('invalid n_layers')
        if type(self.full_visual_tokens) is not int or self.full_visual_tokens < 1:
            raise ValueError('invalid full_visual_tokens')
        last_layer, last_k = 0, self.entry
        for event in self.events:
            if not isinstance(event, tuple) or len(event) != 2:
                raise ValueError('events must be (post_block,new_k) tuples')
            layer, new_k = event
            if type(layer) is not int or not (last_layer < layer < self.n_layers):
                raise ValueError('events must be strictly increasing within decoder')
            if type(new_k) is not int or not (1 <= new_k < last_k):
                raise ValueError('counts must strictly decrease, remain positive')
            last_layer, last_k = layer, new_k

    @classmethod
    def from_segments(cls, name: str, counts: Sequence[int], lengths: Sequence[int],
                      n_layers: int = 32, full_visual_tokens: int = 576) -> 'Schedule':
        if not counts or len(counts) != len(lengths):
            raise ValueError('counts/lengths mismatch')
        if any(type(v) is not int or v < 1 for v in lengths):
            raise ValueError('lengths must be positive integers')
        if sum(lengths) != n_layers:
            raise ValueError('lengths must sum to n_layers')
        cursor, events = 0, []
        for length, next_k in zip(lengths[:-1], counts[1:]):
            cursor += length
            events.append((cursor, next_k))
        return cls(name, counts[0], tuple(events), n_layers, full_visual_tokens)

    def trace(self) -> tuple[int, ...]:
        out, previous, k = [], 0, self.entry
        for block, next_k in (*self.events, (self.n_layers, None)):
            out.extend([k] * (block - previous))
            previous, k = block, next_k
        assert len(out) == self.n_layers
        return tuple(out)

    @property
    def budget(self) -> int:
        return sum(self.trace())

    @property
    def compression_pct(self) -> float:
        return 100.0 * (1.0 - self.budget / (self.full_visual_tokens*self.n_layers))

    def expected_after(self, completed_block: int) -> int | None:
        if type(completed_block) is not int:
            raise ValueError('completed_block must be an integer')
        return dict(self.events).get(completed_block)

    def verify_budget(self, expected_w: int) -> None:
        if self.budget != expected_w:
            raise ValueError(f'{self.name}: W={self.budget}, expected {expected_w}')
        if sum(self.trace()) != expected_w:
            raise AssertionError('independent trace sum disagrees')

    def to_dict(self) -> dict:
        return dict(name=self.name, entry=self.entry, events=[list(e) for e in self.events],
                    n_layers=self.n_layers, full_visual_tokens=self.full_visual_tokens,
                    trace=list(self.trace()), w=self.budget,
                    compression_pct=self.compression_pct)

    @classmethod
    def from_dict(cls, obj: Mapping) -> 'Schedule':
        return cls(name=str(obj['name']), entry=int(obj['entry']),
                   events=tuple(tuple(x) for x in obj['events']),
                   n_layers=int(obj.get('n_layers', 32)),
                   full_visual_tokens=int(obj.get('full_visual_tokens', 576)))


def _unique_ids(ids: Iterable[int], *, name: str) -> tuple[int, ...]:
    items = tuple(ids)
    if any(type(x) is not int or x < 0 for x in items):
        raise ValueError(f'{name}: IDs must be nonnegative python ints')
    if len(items) != len(set(items)):
        raise ValueError(f'{name}: duplicate patch IDs')
    return items


class TokenLedger:
    """Tracks original identities independent of local positions and hidden data.

    Caller provides local indices chosen by host selector and performs all host
    model gather/mask/position/KV operations. DO NOT substitute this object for
    the host's VLM-specific physical-pruning function.
    """
    def __init__(self, original_patch_ids: Iterable[int]):
        self.ids = _unique_ids(original_patch_ids, name='entry')
        if not self.ids:
            raise ValueError('at least one entry patch required')
        self.history = [self.ids]

    def next_keep(self, local_indices: Iterable[int], expected_k: int,
                  *, preserve_local_order: bool = True) -> tuple[int, ...]:
        idxs = tuple(local_indices)
        if type(expected_k) is not int or expected_k < 1 or len(idxs) != expected_k:
            raise ValueError('selector returned wrong count')
        if any(type(i) is not int or i < 0 or i >= len(self.ids) for i in idxs):
            raise ValueError('out-of-range local index')
        if len(set(idxs)) != len(idxs):
            raise ValueError('duplicate local index')
        if preserve_local_order and list(idxs) != sorted(idxs):
            raise ValueError('keep indices must be sorted by original active local order')
        return tuple(self.ids[i] for i in idxs)

    def commit(self, local_indices: Iterable[int], expected_k: int) -> tuple[int, ...]:
        kept = self.next_keep(local_indices, expected_k)
        if len(kept) >= len(self.ids):
            raise ValueError('not a physical pruning step')
        self.ids = kept
        self.history.append(kept)
        return kept

    def indices_of(self, patch_ids: Iterable[int]) -> tuple[int, ...]:
        selected = _unique_ids(patch_ids, name='selected')
        table = {patch_id: i for i, patch_id in enumerate(self.ids)}
        if not set(selected).issubset(table):
            raise ValueError('selected ID not present in active set')
        return tuple(table[x] for x in selected)


@dataclass(frozen=True)
class SurvivalSwap:
    """Identity-only specification for a paired real-model counterfactual.

    A/B must fork AFTER the same prefix state. Their compute budget matches
    only when additional groups have equal cardinality and lifetime. Both
    branches physically keep the same S after the experiment window; their
    hidden states and answers are intentionally permitted to differ.
    """
    s: tuple[int, ...]
    g_a: tuple[int, ...]
    g_b: tuple[int, ...]
    duration: int
    entry: tuple[int, ...]

    @property
    def temporary_k(self) -> int:
        return len(self.s) + len(self.g_a)

    @property
    def delta_token_layers(self) -> int:
        return self.duration * len(self.g_a)

    def verify(self) -> None:
        e = set(_unique_ids(self.entry, name='entry'))
        s = set(_unique_ids(self.s, name='S'))
        a = set(_unique_ids(self.g_a, name='Ga'))
        b = set(_unique_ids(self.g_b, name='Gb'))
        if type(self.duration) is not int or self.duration < 1:
            raise ValueError('duration must be positive')
        if not s or not a or not b:
            raise ValueError('need common survivors and two nonempty groups')
        if len(a) != len(b):
            raise ValueError('Ga/Gb must be equal size')
        if not (s.isdisjoint(a) and s.isdisjoint(b) and a.isdisjoint(b)):
            raise ValueError('Ga, Gb, S must be disjoint')
        if not s | a | b <= e:
            raise ValueError('groups must be from the same entry patch pool')

    def branch_ids(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        self.verify()
        # Same order as original entry patches: no arbitrary position permutation.
        a, b = set(self.s) | set(self.g_a), set(self.s) | set(self.g_b)
        return tuple(x for x in self.entry if x in a), tuple(x for x in self.entry if x in b)


def validate_survival_swap(entry: Iterable[int], common: Iterable[int],
                           group_a: Iterable[int], group_b: Iterable[int],
                           duration: int) -> SurvivalSwap:
    # Materialize the source as a tuple; don't let generators get consumed twice.
    obj = SurvivalSwap(_unique_ids(common, name='S'), _unique_ids(group_a, name='Ga'),
                       _unique_ids(group_b, name='Gb'), duration,
                       _unique_ids(entry, name='entry'))
    obj.verify()
    return obj


def _rank_percentile(scores: Sequence[float]) -> tuple[float, ...]:
    """Deterministic score rank: smallest=0, largest=1; tie by original ID.

    This is an OPTIONAL NEW diagnostic fusion, NOT the verified R3 dual scoring.
    For faithful R3 V1/V2/V3/V5 reuse its original selector callback.
    """
    values = tuple(float(x) for x in scores)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('nonempty finite score vectors required')
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r / max(1, len(values)-1)
    return tuple(ranks)


def combine_rank_scores(pen_scores: Sequence[float], l9_scores: Sequence[float],
                        *, beta: float) -> tuple[float, ...]:
    """Explicit rank-space dual fusion for NEW ablation; independent of model."""
    if not math.isfinite(beta) or not 0. <= beta <= 1.:
        raise ValueError('beta must be within [0,1]')
    if len(pen_scores) != len(l9_scores):
        raise ValueError('pen/L9 lengths differ')
    p, l = _rank_percentile(pen_scores), _rank_percentile(l9_scores)
    return tuple((1-beta)*a+beta*b for a,b in zip(p,l))


def r4_score(gqa: float, pope_f1: float, mme: float, mmbench: float) -> float:
    """Only aggregate already OFFICIALLY SCORED metrics. Never score raw answers here."""
    x = (float(gqa), float(pope_f1), float(mme), float(mmbench))
    if any(not math.isfinite(t) or t < 0 for t in x):
        raise ValueError('invalid official scores')
    return 25. * sum(v/d for v,d in zip(x,(61.9,85.8,1852.,64.1)))


def assert_real_trace(trace: Sequence[int], schedule: Schedule) -> None:
    """Strict per-sample budget proof from *physically computed* visual counts."""
    if len(trace) != schedule.n_layers:
        raise ValueError('wrong number of computed decoder layers')
    if any(type(k) is not int or k < 1 for k in trace):
        raise ValueError('bad physical count')
    if tuple(trace) != schedule.trace():
        mismatch = [(i+1, got, exp) for i,(got,exp) in enumerate(zip(trace,schedule.trace())) if got != exp]
        raise ValueError(f'physical trace differs from configured plan: {mismatch[:6]}')
    if sum(trace) != schedule.budget:
        raise ValueError('W mismatch')
