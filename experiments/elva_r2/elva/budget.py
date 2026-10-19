"""Exact decoder token-layer budgets; events occur AFTER completed blocks.

This module does not estimate task utility or latency. A suffix predictor
must be calibrated for the exact suffix catalog used at deployment.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import math


def integer(x, name, minimum=0):
    if isinstance(x, bool) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(x)


@dataclass(frozen=True)
class Schedule:
    counts: tuple
    lengths: tuple

    def __post_init__(self):
        counts = tuple(integer(k, "count", 1) for k in self.counts)
        lengths = tuple(integer(d, "length", 1) for d in self.lengths)
        if not counts or len(counts) != len(lengths):
            raise ValueError("one positive length per count required")
        if any(a <= b for a, b in zip(counts, counts[1:])):
            raise ValueError("counts must strictly decrease; merge equal stages")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "lengths", lengths)

    @property
    def depth(self):
        return sum(self.lengths)

    @property
    def trace(self):
        return tuple(k for k, d in zip(self.counts, self.lengths) for _ in range(d))

    @property
    def work(self):
        return sum(k * d for k, d in zip(self.counts, self.lengths))

    @property
    def events(self):
        completed, result = 0, []
        for d, k in zip(self.lengths[:-1], self.counts[1:]):
            completed += d
            result.append((completed, k))
        return tuple(result)

    def compression(self, original_tokens):
        n = integer(original_tokens, "original_tokens", 1)
        if self.counts[0] > n:
            raise ValueError("entry exceeds original token count")
        return 1 - Fraction(self.work, n * self.depth)

    def validate_trace(self, actual):
        actual = tuple(integer(k, "trace count", 1) for k in actual)
        if actual != self.trace:
            raise ValueError("actual pre-block visual counts disagree with schedule")

    def proxies(self, text_tokens):
        t = integer(text_tokens, "text_tokens", 0)
        # Dense causal pair count. Protected structural tokens belong in t.
        pairs = sum((t+k)*(t+k+1)//2 for k in self.trace)
        base_pairs = self.depth * t*(t+1)//2
        return dict(token_layers=self.work, visual_square=sum(k*k for k in self.trace),
                    incremental_causal_pairs=pairs-base_pairs)


@dataclass
class Trace:
    """Record counts BEFORE each block, never counts after a pruning hook."""
    budget: int
    depth: int = 32

    def __post_init__(self):
        self.budget = integer(self.budget, "budget", 1)
        self.depth = integer(self.depth, "depth", 1)
        self.counts = []

    @property
    def spent(self):
        return sum(self.counts)

    def before_block(self, block, visual_count):
        block = integer(block, "block", 1)
        k = integer(visual_count, "visual_count", 1)
        if block != len(self.counts)+1 or block > self.depth:
            raise ValueError("blocks must be recorded once in execution order")
        if self.counts and k > self.counts[-1]:
            raise ValueError("deleted tokens cannot reappear")
        if self.spent+k > self.budget:
            raise ValueError("hard token-layer budget exceeded")
        self.counts.append(k)

    def finish(self):
        if len(self.counts) != self.depth or self.spent != self.budget:
            raise ValueError("incomplete trace or exact budget not met")
        return tuple(self.counts)


def feasible_suffixes(*, depth, completed, spent, current_k, budget,
                      boundaries, finals):
    """Enumerate every exact TWO-STAGE continuation in this finite catalog.

    Prefix is already executed. No token resurrection, hidden borrowing,
    or test-batch budget balancing. No global-optimality claim is made.
    """
    depth = integer(depth, "depth", 1)
    completed = integer(completed, "completed")
    spent = integer(spent, "spent")
    current_k = integer(current_k, "current_k", 1)
    budget = integer(budget, "budget", 1)
    if completed >= depth or spent >= budget:
        raise ValueError("no unspent continuation")
    result = set()
    for boundary in boundaries:
        b = integer(boundary, "boundary", 1)
        if not completed < b < depth:
            raise ValueError("boundary must be inside the remaining depth")
        d1, d2 = b-completed, depth-b
        for final in finals:
            f = integer(final, "final", 1)
            numerator = budget-spent-f*d2
            if numerator % d1:
                continue
            middle = numerator//d1
            if f < middle <= current_k:
                result.add(Schedule((middle, f), (d1, d2)))
    return tuple(sorted(result, key=lambda p: (p.lengths, p.counts)))


def choose_suffix(catalog, predicted_gains, *, anchor=0, threshold=0.):
    """Single-path choice; zero/negative/uncertain gains keep the anchor.

    Gains must be relative to anchor, from a suffix-action estimator.
    A group-survival estimator is NOT a valid substitute.
    """
    if not catalog or len(catalog) != len(predicted_gains):
        raise ValueError("one prediction per feasible suffix required")
    anchor = integer(anchor, "anchor")
    if anchor >= len(catalog) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid anchor/threshold")
    if len({(p.work, p.depth) for p in catalog}) != 1:
        raise ValueError("suffixes must have identical remaining work and depth")
    gains = tuple(float(v) for v in predicted_gains)
    if not all(math.isfinite(v) for v in gains) or gains[anchor] != 0.:
        raise ValueError("finite, anchor-relative predictions required")
    best = max(range(len(gains)), key=lambda i: gains[i])
    return anchor if gains[best] <= threshold else best

