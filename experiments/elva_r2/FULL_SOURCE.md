# ELVA R2 完整 Python 源码

从本次恢复的实际源文件逐文件展开；完整配置、文档、验证记录与原文件均在 ZIP 中。以源文件为执行依据，本展开稿未删节任何 Python 文件。

## `elva/__init__.py`

SHA-256: `c18d728828c9de01c9f23b1bd4e4e60cbb3a6a6aa5a3b1271e9e13db3fd2a56c`

````python
"""ELVA R2: CPU reference and source-bound VLM integration contracts."""
from .budget import Schedule, Trace, feasible_suffixes, choose_suffix

````

## `elva/benchmark.py`

SHA-256: `ffb01483d54f9dffa0a19f53383c6275df819d78aa6206b2aa2fc3a44f68cd81`

````python
"""Paired end-to-end FIRST TOKEN timing; real VLM is a bound callback.

Input preparation belongs inside run_request. Dedicated measurement device.
No CUDA-event-only result is relabeled as user-observable wall TTFT.
"""
import math
import random
import time
import numpy as np


def paired_ttft(samples, methods, run_request, synchronize, *, rounds=3, warmup=3, seed=19):
    if len(methods) != 2 or methods[0] == methods[1] or not samples or type(rounds) is not int or rounds < 1:
        raise ValueError("two distinct methods, samples, positive rounds required")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be nonnegative")
    for i in range(warmup):
        for method in methods:
            run_request(samples[i % len(samples)], method)
    synchronize()
    rng, rows = random.Random(seed), []
    for repeat in range(rounds):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for i in order:
            # Balanced order over rounds/input order, avoiding all-A then all-B.
            local = methods if (i+repeat) % 2 == 0 else methods[::-1]
            for method in local:
                synchronize()
                start = time.perf_counter_ns()
                first_token = run_request(samples[i], method)
                synchronize()
                elapsed = (time.perf_counter_ns()-start)/1e6
                if first_token is None or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("callback must emit an actual first token")
                rows.append(dict(sample=i, repeat=repeat, method=method, wall_ttft_ms=elapsed,
                                 first_token=str(first_token)))
    return rows


def summarize(rows):
    output = {}
    for method in sorted(set(row["method"] for row in rows)):
        values = np.array([row["wall_ttft_ms"] for row in rows if row["method"] == method], float)
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("invalid timing rows")
        output[method] = dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                              p90_ms=float(np.quantile(values, .9)), observations=len(values))
    return output
````

## `elva/budget.py`

SHA-256: `ff2831cf2bd68ceec1899d83f87812cf9f5027233e7b3324e9b5394073536f83`

````python
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

````

## `elva/capture.py`

SHA-256: `aa7991ac4f6b6b6dab1bbfb13a6036545270c8990589942024e5a67e6751cf22`

````python
"""Single-forward layer capture using native module hooks, without attentions.

Pass the actual source-bound ViT block list. Completed block 9 -> blocks[8].
For a D-block CLIP, hidden_states[-2] -> output of completed block D-1.
This is a capture utility, not a model-specific architecture resolver.
"""
from .budget import integer


class LayerCapture:
    def __init__(self, blocks, *, early_completed=9, pen_completed=None):
        self.blocks = list(blocks)
        self.early = integer(early_completed, "early_completed", 1)
        self.pen = len(self.blocks)-1 if pen_completed is None else integer(pen_completed, "pen_completed", 1)
        if not self.blocks or not 1 <= self.early <= len(self.blocks) or not 1 <= self.pen <= len(self.blocks):
            raise ValueError("requested layer is outside encoder")
        self.values, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("capture already attached")
        self.values.clear()
        def hook_for(completed):
            def hook(module, args, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if completed in self.values:
                    raise RuntimeError("unexpected repeated forward; clear capture per request")
                # Clone only the requested layers; record its cost in TTFT.
                self.values[completed] = value.detach().clone() if hasattr(value, "detach") else value.copy()
            return hook
        for completed in sorted({self.early, self.pen}):
            self.handles.append(self.blocks[completed-1].register_forward_hook(hook_for(completed)))
        return self

    def get(self):
        if not {self.early, self.pen}.issubset(self.values):
            raise RuntimeError("required layer did not execute")
        return self.values[self.early], self.values[self.pen]

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
````

## `elva/counterfactual.py`

SHA-256: `1f6d092a57593cf909f3a41db8ce2e8345e70ff4f585933f9866bc5cdce62fc6`

````python
"""True temporary-survival swaps, with native and support-aligned cache modes.

Model-specific execution belongs to a source-bound adapter, never a guessed
monkey patch. Both arms remove the compared groups before the shared suffix.
"""
from dataclasses import dataclass
import math
import numpy as np
from .budget import integer
from .evidence import ids_array


ESTIMAND = "transient_survival_swap_v1"


@dataclass(frozen=True)
class Swap:
    begin: int                 # completed blocks at the shared prefix
    end: int                   # completed blocks when both groups disappear
    common: tuple              # shared support DURING the window
    group_a: tuple
    group_b: tuple
    final: tuple               # identical support AFTER the window

    def __post_init__(self):
        begin, end = integer(self.begin, "begin"), integer(self.end, "end", 1)
        if end <= begin:
            raise ValueError("positive survival window required")
        for name in ("common", "group_a", "group_b", "final"):
            object.__setattr__(self, name, tuple(map(int, ids_array(getattr(self, name)))))
        c, a, b, f = map(set, (self.common, self.group_a, self.group_b, self.final))
        if c&a or c&b or a&b or len(a) != len(b) or not f.issubset(c):
            raise ValueError("disjoint equal groups and final subset of common required")

    @property
    def window_k(self):
        return len(self.common)+len(self.group_a)


def _native_equal(value_a, value_b, message):
    if value_a != value_b:
        raise ValueError(message)


def _validate_view(view, swap, prefix_view):
    required = {"prefix_fingerprint", "completed", "active_ids", "kv_support",
                "protected_positions", "visual_positions"}
    if not required.issubset(view):
        raise ValueError("adapter inspection missing mandatory fields")
    _native_equal(view["prefix_fingerprint"], prefix_view["prefix_fingerprint"], "prefix identity changed")
    _native_equal(view["protected_positions"], prefix_view["protected_positions"], "text/structural positions changed")
    expected_positions = {i: prefix_view["visual_positions"][i] for i in swap.final}
    _native_equal(view["visual_positions"], expected_positions, "original visual positions changed")
    _native_equal(view["completed"], swap.end, "wrong join boundary")
    _native_equal(tuple(sorted(view["active_ids"])), tuple(sorted(swap.final)), "final identity mismatch")


def run_swap(adapter, prefix, swap, *, image_key, question_key, mode="native"):
    """Adapter methods: inspect/fork/prune/advance/align_cache/score_suffix.

    prefix contains prompt-only states. score_suffix computes answer-only
    token-mean NLL and real generation using labels ONLY after decisions.
    Native labels include access to earlier-layer KV during later decoding.
    Aligned-support labels remove extra visual KVs from every computed layer;
    surviving text/visual values intentionally remain branch-specific.
    """
    if mode not in ("native", "aligned_support"):
        raise ValueError("invalid cache intervention")
    before = adapter.inspect(prefix)
    if before["completed"] != swap.begin:
        raise ValueError("prefix is not at the intervention boundary")
    active = set(before["active_ids"])
    if not set(swap.common+swap.group_a+swap.group_b).issubset(active):
        raise ValueError("group contains deleted/unknown token")
    prefix_hash = before["prefix_fingerprint"]
    arms, views = [], []
    for group in (swap.group_a, swap.group_b):
        state = adapter.fork(prefix)
        if state is prefix:
            raise ValueError("fork aliases shared prefix")
        adapter.prune(state, tuple(sorted(swap.common+group)))
        adapter.advance(state, swap.end)
        adapter.prune(state, tuple(sorted(swap.final)))
        if mode == "aligned_support":
            adapter.align_cache(state, tuple(sorted(swap.final)))
        view = adapter.inspect(state)
        _validate_view(view, swap, before)
        views.append(view)
        outcome = adapter.score_suffix(state)
        if not {"nll", "answer_tokens", "trace", "backend", "generated"}.issubset(outcome):
            raise ValueError("incomplete real suffix outcome")
        if not math.isfinite(outcome["nll"]) or outcome["nll"] < 0 or outcome["answer_tokens"] < 1:
            raise ValueError("invalid answer-only NLL")
        arms.append(outcome)
        if adapter.inspect(prefix) != before:
            raise ValueError("branch mutated shared prefix; deep-copy cache/metadata")
    _native_equal(arms[0]["trace"], arms[1]["trace"], "not an equal-work swap")
    _native_equal(arms[0]["backend"], arms[1]["backend"], "attention backend differs")
    _native_equal(arms[0]["answer_tokens"], arms[1]["answer_tokens"], "answer normalization differs")
    if mode == "aligned_support":
        _native_equal(views[0]["kv_support"], views[1]["kv_support"], "cache supports differ at join")
        expected = set(swap.final)
        if any(set(layer) != expected for layer in views[0]["kv_support"]):
            raise ValueError("extra/missing visual cache IDs after support alignment")
    trace = tuple(arms[0]["trace"])
    if len(trace) < swap.end or trace[swap.begin:swap.end] != (swap.window_k,)*(swap.end-swap.begin):
        raise ValueError("adapter trace does not execute the intended survival window")
    return dict(schema=ESTIMAND, image_key=str(image_key), question_key=str(question_key),
                cache_mode=mode, prefix_fingerprint=prefix_hash,
                begin=swap.begin, end=swap.end, final_ids=list(swap.final),
                common=list(swap.common), group_a=list(swap.group_a), group_b=list(swap.group_b),
                work=sum(trace), trace=list(trace), nll_a=arms[0]["nll"], nll_b=arms[1]["nll"],
                delta=arms[1]["nll"]-arms[0]["nll"],
                generated_a=arms[0]["generated"], generated_b=arms[1]["generated"],
                backend=arms[0]["backend"],
                note="positive delta favors group_a; offline label, not online input")


def answer_nll(logits, labels, answer_mask):
    """Standard CAUSAL shift: logits[t] predicts labels[t+1]. No fake EOS.

    Single full-sequence CPU reference. When prompt is compacted, routing
    and masks must refer to the compacted sequence. Token-mean loss only.
    """
    z = np.asarray(logits, dtype=float)
    y, mask = np.asarray(labels), np.asarray(answer_mask)
    if z.ndim != 2 or len(z) < 2 or y.shape != (len(z),) or mask.shape != y.shape:
        raise ValueError("aligned [T,V] logits and [T] labels/mask required")
    if y.dtype.kind not in "iu" or mask.dtype != np.bool_ or not np.isfinite(z).all():
        raise ValueError("integer labels, boolean answer mask and finite logits required")
    valid = mask[1:] & (y[1:] != -100)
    if not valid.any():
        raise ValueError("no answer targets")
    target = y[1:][valid]
    if target.min() < 0 or target.max() >= z.shape[1]:
        raise ValueError("answer label outside vocabulary")
    values = z[:-1][valid]
    shift = values.max(-1)
    lse = shift+np.log(np.exp(values-shift[:, None]).sum(-1))
    return float((lse-values[np.arange(len(target)), target]).mean())

````

## `elva/evidence.py`

SHA-256: `4383f02f50063cf7466c3cd19fa20d5d3b83e0a712218acfcd89df7f97d978d4`

````python
"""Identity-safe Stage1 selection and attention/rank reference functions.

COVER/QED are intentionally NOT approximated. Bind original source callbacks.
NumPy reference only; moving CUDA tensors to CPU is not a deployment path.
"""
import numpy as np


def indices(value, n, size=None):
    a = np.asarray(value)
    if (a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0
            or len(np.unique(a)) != len(a) or a.min() < 0 or a.max() >= n
            or (size is not None and len(a) != size)):
        raise ValueError("unique nonempty integer active-local indices required")
    return a.astype(np.int64, copy=False)


def ids_array(value):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0 or len(np.unique(a)) != len(a) or a.min() < 0:
        raise ValueError("unique nonnegative original IDs required")
    return a.astype(np.int64, copy=False)


def finite(value, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("finite nonempty array required")
    return a


def unit(x):
    x = finite(x, 2).astype(np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x/np.maximum(norm, 1e-12)


def topk(scores, k):
    s = finite(scores, 1)
    if type(k) is not int or not 0 < k <= len(s):
        raise ValueError("invalid K")
    return np.sort(np.argsort(-s, kind="stable")[:k])


def select_stage1(pen, pen_scores, ids, k, *, guide="pen", early=None,
                  early_scores=None, early_ids=None, beta=.5, cover=None,
                  use_cover=False, context=None, frozen_keep=None, score_guide="same"):
    """Guidance changes selection only. Payload is ALWAYS raw pen[keep].

    Layer scores must use the same bound saliency rule and normalization.
    Patch IDs include image/crop identity. Returned local keep is spatially
    ordered; projector input dimension and vector values are unchanged.
    """
    x, a, pid = finite(pen, 2), finite(pen_scores, 1), ids_array(ids)
    if x.dtype.kind != "f" or a.shape != (len(x),) or len(pid) != len(x) or (a < 0).any():
        raise ValueError("floating payload and aligned nonnegative saliency required")
    if type(k) is not int or not 0 < k <= len(x):
        raise ValueError("invalid K")
    if guide not in ("pen", "l9", "dual") or not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("invalid guidance")
    sg = guide if score_guide == "same" else score_guide
    if sg not in ("pen", "l9", "dual"):
        raise ValueError("invalid score guide")
    needs_early = (guide == "l9" or (guide == "dual" and beta > 0)
                   or sg == "l9" or (sg == "dual" and beta > 0))
    if needs_early:
        e, b = finite(early, 2), finite(early_scores, 1)
        eid = ids_array(early_ids)
        if e.shape != x.shape or b.shape != a.shape or not np.array_equal(pid, eid) or (b < 0).any():
            raise ValueError("cross-layer feature/score/ID alignment failed")
    if guide == "pen" or (guide == "dual" and beta == 0):
        metric, scores = x, a
    elif guide == "l9" or beta == 1:
        metric, scores = e, b
    else:
        metric = np.concatenate((np.sqrt(1-beta)*unit(x), np.sqrt(beta)*unit(e)), axis=1)
    scores = a if sg == "pen" or (sg == "dual" and beta == 0) else (
        b if sg == "l9" or beta == 1 else (1-beta)*a+beta*b)
    if frozen_keep is not None:
        keep = indices(frozen_keep, len(x), k)
    elif use_cover:
        if cover is None:
            raise RuntimeError("BLOCKED_SOURCE: bind original COVER")
        keep = indices(cover(metric_features=metric.copy(), scores=scores.copy(),
                             patch_ids=pid.copy(), k=k, context=context), len(x), k)
    else:
        keep = topk(scores, k)
    keep = np.sort(keep)
    return dict(payload=x[keep].copy(), keep=keep, ids=pid[keep].copy(),
                metric=metric.copy(), scores=scores.copy())


def question_visual_mass(logits, allowed, visual_columns):
    """Partial question QK only, AFTER RoPE/scaling; [Q,T] or [H,Q,T].

    Denominator covers ALL causal-visible keys; answer rows are forbidden
    by the caller's prompt-only query routing, not guessed here.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim not in (2, 3) or min(z.shape) == 0:
        raise ValueError("logits must be [Q,T] or [H,Q,T]")
    raw_mask = np.asarray(allowed)
    if raw_mask.dtype != np.bool_:
        raise ValueError("explicit boolean causal/padding mask required")
    mask = np.broadcast_to(raw_mask, z.shape)
    if not mask.any(-1).all() or not np.isfinite(z[mask]).all():
        raise ValueError("invalid visible row")
    cols = indices(visual_columns, z.shape[-1])
    z = np.where(mask, z, -np.inf)
    p = np.exp(z-z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p[..., cols].mean(axis=tuple(range(z.ndim-1)))


def average_ranks(scores):
    """Larger response -> larger rank. Equal scores get equal average rank."""
    s = finite(scores, 1).astype(float)
    order = np.argsort(s, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(s[order]) != 0)+1]
    stops = np.r_[starts[1:], len(s)]
    ranks = np.empty(len(s), dtype=float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start+stop-1)/2
    return ranks/max(1, len(s)-1)


def rank_drift(current, active_ids, previous, previous_ids, weight=.6):
    """User-specified RDRIFT reference, NOT claimed byte-identical to legacy.

    Rank both responses on the SAME current candidate universe. Align history
    by original IDs, never compare post-prune ranks with full-pool ranks.
    """
    now, old = finite(current, 1), finite(previous, 1)
    active, history = ids_array(active_ids), ids_array(previous_ids)
    if len(now) != len(active) or len(old) != len(history) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("invalid history/weight")
    lookup = {int(v): i for i, v in enumerate(history)}
    if any(int(v) not in lookup for v in active):
        raise ValueError("missing history; do not silently substitute current")
    matched = old[[lookup[int(v)] for v in active]]
    rn, ro = average_ranks(now), average_ranks(matched)
    return weight*rn+(1-weight)*(rn-ro)


def compact_sequence(hidden, original_positions, visual_ids, keep_ids):
    """CPU reference only: retain ALL text/structural rows (-1), original RoPE positions."""
    h = finite(hidden, 2)
    p, v = np.asarray(original_positions), np.asarray(visual_ids)
    keep = ids_array(keep_ids)
    if p.shape != (len(h),) or v.shape != p.shape or p.dtype.kind not in "iu" or v.dtype.kind not in "iu":
        raise ValueError("aligned sequence positions/visual IDs required")
    if len(np.unique(p)) != len(p) or (np.diff(p) <= 0).any() or (v < -1).any():
        raise ValueError("strict original positions and -1 for protected rows required")
    current = v[v >= 0]
    if len(np.unique(current)) != len(current) or not set(keep).issubset(set(current)):
        raise ValueError("duplicate or resurrected visual IDs")
    rows = np.flatnonzero((v < 0) | np.isin(v, keep))
    return h[rows].copy(), p[rows].copy(), v[rows].copy(), rows
````

## `elva/integration.py`

SHA-256: `e6face2184924c96dd4e4d5b003c911f82610abeaa0844bb4de58bb6c720d053`

````python
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
````

## `elva/learning.py`

SHA-256: `c399b0aca041c9b4683996be80641fd9657ab858799f8c31ac13553a5ee254a1`

````python
"""Optional diagnostic signed-gain regressor for NEW transient-swap labels.

It is deliberately small. A working synthetic regression is not evidence
that VLM survival gains are predictable. No old E020 weights are accepted.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import ESTIMAND
from .evidence import finite, indices


def split_image(image_key, salt="elva-r2-frozen-split"):
    """Split by canonical decoded-image identity, not question/filename."""
    if not isinstance(image_key, str) or not image_key:
        raise ValueError("canonical image key required")
    value = int(hashlib.sha256((salt+image_key).encode()).hexdigest()[:16], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def pair_features(token_features, common, group_a, group_b, *, begin, end, depth):
    """Features only from prompt and states available BEFORE this decision.

    Caller supplies aligned observables (current mass, rank drift, ViT CLS,
    layer-specific coverage descriptors, etc.). Do not feed NLL, future
    states, teacher-forced answer queries, or oracle-selected final IDs.
    Reversing groups negates the vector exactly, including context terms.
    """
    x = finite(token_features, 2).astype(float)
    c, a, b = (indices(v, len(x)) for v in (common, group_a, group_b))
    if set(c)&set(a) or set(c)&set(b) or set(a)&set(b) or len(a) != len(b):
        raise ValueError("disjoint equal groups required")
    if not 0 <= begin < end <= depth:
        raise ValueError("invalid observation/window depth")
    delta = x[a].mean(0)-x[b].mean(0)
    variance = x[a].var(0)-x[b].var(0)
    context = x[c].mean(0)
    return np.r_[delta, variance, delta*context, delta*(end-begin)/depth]


@dataclass
class PairModel:
    scale: np.ndarray
    weights: np.ndarray
    cache_mode: str
    feature_names: tuple
    catalog_hash: str
    stage: str = "decoder_transient_swap"

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("finite features with trained schema required")
        return (x/self.scale)@self.weights

    def save(self, path):
        value = dict(schema=ESTIMAND, stage=self.stage, cache_mode=self.cache_mode,
                     feature_names=list(self.feature_names), catalog_hash=self.catalog_hash,
                     scale=self.scale.tolist(), weights=self.weights.tolist())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path, *, cache_mode, catalog_hash, feature_names):
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        if (value.get("schema") != ESTIMAND or value.get("stage") != "decoder_transient_swap"
                or value.get("cache_mode") != cache_mode or value.get("catalog_hash") != catalog_hash
                or value.get("feature_names") != list(feature_names)):
            raise ValueError("incompatible weights/estimand/catalog; never reuse old E020 head")
        scale, weights = np.asarray(value["scale"], float), np.asarray(value["weights"], float)
        if (scale.ndim != 1 or len(scale) == 0 or weights.shape != scale.shape
                or len(feature_names) != len(scale) or not np.isfinite(scale).all()
                or not np.isfinite(weights).all() or (scale <= 0).any()):
            raise ValueError("invalid model arrays")
        return cls(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def fit_pairs(records, *, feature_names, catalog_hash, cache_mode, ridge=1.):
    """Train split only, no intercept/mean subtraction; antisymmetry preserved."""
    if not np.isfinite(ridge) or ridge <= 0 or not records:
        raise ValueError("positive ridge and nonempty real records required")
    if (not feature_names or len(set(feature_names)) != len(feature_names)
            or not catalog_hash or cache_mode not in ("native", "aligned_support")):
        raise ValueError("nonempty unique feature schema and valid catalog/cache mode required")
    xs, ys = [], []
    images = set()
    for record in records:
        if (record.get("schema") != ESTIMAND or record.get("cache_mode") != cache_mode
                or record.get("catalog_hash") != catalog_hash
                or record.get("feature_names") != list(feature_names)
                or record.get("feature_cutoff") != record.get("begin")
                or split_image(record["image_key"]) != "train"):
            raise ValueError("invalid schema/cache/catalog/causal cutoff or split leakage")
        if record.get("fixture", False):
            raise ValueError("synthetic fixtures cannot train a deployable VLM model")
        x = np.asarray(record["x"], float)
        y = float(record["delta"])
        if x.shape != (len(feature_names),) or not np.isfinite(x).all() or not np.isfinite(y):
            raise ValueError("nonfinite or mismatched training data")
        xs.append(x)
        ys.append(y)
        images.add(record["image_key"])
    if len(images) < 2:
        raise ValueError("at least two independent training images required")
    x, y = np.asarray(xs), np.asarray(ys)
    scale = np.sqrt((x*x).mean(0))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = x/scale
    weights = np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]), z.T@y)
    return PairModel(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def choose_swap(predicted_gains, *, threshold):
    """One complete frozen candidate catalog, index 0 is the no-op."""
    gains = finite(predicted_gains, 1).astype(float)
    if gains[0] != 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("zero no-op gain and calibrated nonnegative threshold required")
    best = int(np.argmax(gains))
    return best if gains[best] > threshold else 0


def calibrate_threshold(predicted, true_gains, noop_differences):
    """VALIDATION images only, equal-work actions incl. true no-op column 0.

    Accepts complete [states, actions] catalogs. Threshold candidates use
    predicted maxima, not individual-action thresholds. Actual VLM generation
    confirmation is a separate requirement even if NLL gain is positive.
    """
    p, y = finite(predicted, 2), finite(true_gains, 2)
    noise = finite(noop_differences, 1)
    if p.shape != y.shape or (p[:, 0] != 0).any() or (y[:, 0] != 0).any():
        raise ValueError("complete matched catalogs with zero no-op column required")
    floor = float(np.quantile(np.abs(noise), .95))
    maxima = np.maximum(0., p.max(1))
    candidates = np.unique(np.r_[floor, maxima[maxima >= floor]])
    best = None
    for threshold in candidates:
        actions = np.array([choose_swap(row, threshold=float(threshold)) for row in p])
        gains = y[np.arange(len(y)), actions]
        score = float(gains.mean())
        key = (score, float(threshold))  # conservative tie: fewer accepted actions
        if best is None or key > best[0]:
            best = key, actions
    return dict(threshold=best[0][1], validation_gain=best[0][0],
                accepted=int((best[1] != 0).sum()), states=len(p),
                noise_floor=floor,
                status="CANDIDATE" if best[0][0] > floor and (best[1] != 0).any() else "NO_SUPPORTED_DECISIONS")
````

## `elva/policy.py`

SHA-256: `82611baf96ca01c0d17da639d278a0870aea43cd301274ea29fe2324b3d91758`

````python
"""Single-swap lifetime policy; offline and online share one finite catalog.

Selected groups live only to end, then BOTH possible groups are discarded.
This is distinct from old final-identity replacement. NumPy reference only.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import Swap, run_swap
from .evidence import finite, ids_array, indices, topk
from .learning import pair_features, choose_swap


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Catalog:
    baseline: tuple               # active-local baseline indices
    final: tuple                  # active-local final indices, fixed before choice
    swaps: tuple                  # (common, inserted, removed), active-local
    rule_hash: str                 # shared across images, used by model manifest
    action_hash: str               # particular identities in this sample
    window: tuple
    active_ids: tuple


def propose(scores, active_ids, baseline_keep, final_k, *, group_size=4, limit=16,
            begin=10, end=20, depth=32, proposal_rule="BIND_NATIVE_RULE"):
    """At most limit TOTAL actions incl. no-op; no resurrection or answer use.

    scores is a proposal ranking (native Q_ATTN or RDRIFT), not a utility
    certificate. Final anchors are chosen within baseline NOW, never from a
    future answer-dependent reference. Labels must cover this exact catalog.
    """
    s, pid = finite(scores, 1), ids_array(active_ids)
    if len(s) != len(pid) or type(final_k) is not int or type(group_size) is not int or type(limit) is not int:
        raise ValueError("aligned scores/IDs and integer sizes required")
    base = indices(baseline_keep, len(pid))
    if not 0 < final_k <= len(base) or group_size < 1 or limit < 1:
        raise ValueError("invalid final/group/catalog sizes")
    if not all(type(v) is int for v in (begin, end, depth)) or not 0 <= begin < end <= depth:
        raise ValueError("invalid survival window")
    final = np.sort(base[topk(s[base], final_k)])
    removable = np.setdiff1d(base, final)
    outside = np.setdiff1d(np.arange(len(s)), base)
    # Rank tie-breaking is stable in active sequence order.
    removable = removable[np.argsort(s[removable], kind="stable")]
    outside = outside[np.argsort(-s[outside], kind="stable")]
    chunks = lambda a: [a[i:i+group_size] for i in range(0, len(a)-group_size+1, group_size)]
    candidates = []
    for d in chunks(removable):
        for inserted in chunks(outside):
            common = np.setdiff1d(base, d)
            if len(common) == 0:
                continue
            proxy = float(s[inserted].sum()-s[d].sum())
            candidates.append((proxy, tuple(common), tuple(inserted), tuple(d)))
    candidates.sort(key=lambda a: (-a[0], a[2], a[3]))
    swaps = tuple(row[1:] for row in candidates[:limit-1])
    rule = dict(version="single_transient_swap_catalog_v1", group_size=group_size,
                limit=limit, final_k=final_k, middle_k=len(base),
                begin=begin, end=end, depth=depth, proposal_rule=proposal_rule,
                grouping="stable-score-disjoint-chunks", final="current-baseline-topk")
    base, final = tuple(sorted(map(int, base))), tuple(map(int, final))
    identity = dict(ids=pid.tolist(), baseline=base, final=final, swaps=swaps)
    # Convert NumPy index scalars before canonical JSON hashing.
    swaps = tuple(tuple(tuple(map(int, part)) for part in action) for action in swaps)
    identity["swaps"] = swaps
    return Catalog(base, final, swaps, hash_value(rule), hash_value(identity),
                   (begin, end, depth), tuple(map(int, pid)))


def feature_catalog(catalog, token_features, *, begin, end, depth):
    if (begin, end, depth) != catalog.window:
        raise ValueError("survival window differs from trained catalog")
    x = finite(token_features, 2)
    zero = np.zeros(x.shape[1]*4)
    return np.array([zero]+[pair_features(x, c, inserted, removed,
                                         begin=begin, end=end, depth=depth)
                            for c, inserted, removed in catalog.swaps])


def decide(catalog, token_features, model, *, begin, end, depth, threshold):
    if (model.catalog_hash != catalog.rule_hash or model.stage != "decoder_transient_swap"
            or model.cache_mode != "native"):
        raise ValueError("model/catalog mismatch")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    predicted = np.asarray(model.predict(features), float)
    action = choose_swap(predicted, threshold=threshold)
    if action == 0:
        keep = catalog.baseline
    else:
        common, inserted, _ = catalog.swaps[action-1]
        keep = tuple(sorted(common+inserted))
    # Caller must force catalog.final at end. Native re-selection there would
    # change the estimand and invalidate this head's deployment contract.
    return dict(action=action, keep_local=keep,
                keep_ids=tuple(catalog.active_ids[i] for i in keep),
                final_ids=tuple(catalog.active_ids[i] for i in catalog.final),
                predicted_gain=float(predicted[action]), action_hash=catalog.action_hash)


def collect_catalog(adapter, prefix, catalog, active_ids, token_features, *,
                    begin, end, depth, image_key, question_key, feature_names,
                    cache_mode="native"):
    pid = ids_array(active_ids)
    if (tuple(map(int, pid)) != catalog.active_ids
            or tuple(map(int, pid)) != tuple(adapter.inspect(prefix)["active_ids"])
            or len(token_features) != len(pid)):
        raise ValueError("prefix, feature rows and proposal original IDs must have identical order")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    if len(feature_names) != features.shape[1]:
        raise ValueError("feature names do not match policy input")
    rows = []
    state_key = hash_value(dict(prefix=adapter.inspect(prefix)["prefix_fingerprint"],
                                image=image_key, question=question_key, begin=begin, end=end,
                                depth=depth, final=[int(pid[i]) for i in catalog.final],
                                action_hash=catalog.action_hash, cache_mode=cache_mode))
    for i, (common, inserted, removed) in enumerate(catalog.swaps, 1):
        original = lambda local: tuple(int(pid[j]) for j in local)
        swap = Swap(begin, end, original(common), original(inserted), original(removed), original(catalog.final))
        row = run_swap(adapter, prefix, swap, image_key=image_key, question_key=question_key, mode=cache_mode)
        row.update(state_key=state_key, catalog_hash=catalog.rule_hash,
                   action_hash=catalog.action_hash, catalog_size=len(catalog.swaps)+1,
                   action_index=i, x=features[i].tolist(), feature_names=list(feature_names),
                   feature_cutoff=begin, fixture=bool(getattr(adapter, "is_fixture", True)))
        rows.append(row)
    return rows
````

## `elva/torch_ops.py`

SHA-256: `c0b20e07f681453834361c49265d8fb984d0afe4be496925a90eda88136bb99a`

````python
"""CUDA-resident Stage1 path. Server parity test is mandatory before ACC.

No attention backend is changed here. Existing COVER stays source-bound.
CPU reference in evidence.py is the tested mathematical specification.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_stage1(pen, pen_scores, ids, k, *, guide="pen", early=None,
                  early_scores=None, early_ids=None, beta=.5, cover=None,
                  use_cover=False, context=None, frozen_keep=None, debug=True,
                  score_guide="same"):
    if pen.ndim != 2 or min(pen.shape) == 0 or not pen.is_floating_point():
        raise ValueError("floating raw pen [N,D] required")
    n = len(pen)
    if (pen_scores.shape != (n,) or ids.shape != (n,) or ids.dtype != torch.long
            or pen_scores.device != pen.device or ids.device != pen.device
            or not pen_scores.is_floating_point() or type(k) is not int or not 0 < k <= n):
        raise ValueError("aligned source saliency/IDs and valid K required")
    if guide not in ("pen", "l9", "dual") or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("invalid guide/beta")
    sg = guide if score_guide == "same" else score_guide
    if sg not in ("pen", "l9", "dual"):
        raise ValueError("invalid score guide")
    needs_early = (guide == "l9" or (guide == "dual" and beta > 0)
                   or sg == "l9" or (sg == "dual" and beta > 0))
    if needs_early:
        if (early is None or early.shape != pen.shape or early.device != pen.device
                or not early.is_floating_point() or early_scores is None
                or early_scores.shape != (n,) or early_scores.device != pen.device
                or not early_scores.is_floating_point() or early_ids is None
                or early_ids.shape != ids.shape or early_ids.dtype != torch.long
                or early_ids.device != pen.device):
            raise ValueError("aligned captured early features/scores/IDs required")
    if debug:
        tensors = [pen, pen_scores]+([early, early_scores] if needs_early else [])
        if (any(not bool(torch.isfinite(x).all()) for x in tensors)
                or bool((pen_scores < 0).any()) or ids.unique().numel() != n or bool((ids < 0).any())
                or (needs_early and (not torch.equal(ids, early_ids) or bool((early_scores < 0).any())))):
            raise ValueError("nonfinite/negative/misaligned evidence")
    if guide == "pen" or (guide == "dual" and beta == 0):
        metric, scores = pen, pen_scores
    elif guide == "l9" or beta == 1:
        metric, scores = early, early_scores
    else:
        metric = torch.cat((math.sqrt(1-beta)*F.normalize(pen.float(), dim=-1, eps=1e-12),
                            math.sqrt(beta)*F.normalize(early.float(), dim=-1, eps=1e-12)), dim=-1)
    scores = pen_scores if sg == "pen" or (sg == "dual" and beta == 0) else (
        early_scores if sg == "l9" or beta == 1 else (1-beta)*pen_scores.float()+beta*early_scores.float())
    if frozen_keep is not None:
        keep = frozen_keep
    elif use_cover:
        if cover is None:
            raise RuntimeError("BLOCKED_SOURCE: bind original COVER")
        keep = cover(metric_features=metric.clone(), scores=scores.clone(),
                     patch_ids=ids.clone(), k=k, context=context)
    else:
        keep = torch.argsort(scores, descending=True, stable=True)[:k]
    if not isinstance(keep, torch.Tensor) or keep.shape != (k,) or keep.dtype != torch.long or keep.device != pen.device:
        raise ValueError("native selector must return device-local long [K]")
    if debug and (keep.unique().numel() != k or int(keep.min()) < 0 or int(keep.max()) >= n):
        raise ValueError("invalid/duplicate selected local indices")
    keep = keep.sort().values
    return dict(payload=pen.index_select(0, keep), keep=keep,
                ids=ids.index_select(0, keep), metric=metric, scores=scores)


@torch.no_grad()
def question_visual_mass(logits, allowed, visual_columns):
    if logits.ndim not in (2, 3) or min(logits.shape) == 0 or allowed.dtype != torch.bool:
        raise ValueError("partial [Q,T]/[H,Q,T] logits and boolean visible mask required")
    if allowed.device != logits.device or visual_columns.device != logits.device or visual_columns.dtype != torch.long:
        raise ValueError("device-local masks/column indices required")
    mask = torch.broadcast_to(allowed, logits.shape)
    # Validate once in server preflight; this function deliberately makes its
    # synchronization cost visible in end-to-end timing.
    if (not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all())
            or visual_columns.ndim != 1 or visual_columns.numel() == 0
            or visual_columns.unique().numel() != visual_columns.numel()
            or int(visual_columns.min()) < 0 or int(visual_columns.max()) >= logits.shape[-1]):
        raise ValueError("invalid visible rows/visual indices")
    p = logits.float().masked_fill(~mask, -torch.inf).softmax(-1)
    return p.index_select(-1, visual_columns).mean(dim=tuple(range(logits.ndim-1)))
````

## `elva/toy_adapter.py`

SHA-256: `2bc97d9dcf0470895ca0afc143a484e89654a1eb35cbb00fd39935bd3c9fdadd`

````python
"""Random causal decoder fixture. No pretrained VLM, no benchmark claim.

Runs prompt prefill and multi-token cached teacher forcing. Visual IDs have
no relation to positions; earlier caches survive normal hidden-row pruning.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .evidence import compact_sequence


class ToyAdapter:
    is_fixture = True
    def __init__(self, depth=8, visual=12, text=3, dim=12, vocab=9, seed=7):
        self.depth, self.vocab = depth, vocab
        r = np.random.default_rng(seed)
        self.weights = r.normal(0, .15, (depth, 3, dim, dim))
        self.output = r.normal(0, .3, (dim, vocab))
        self.embeddings = r.normal(0, .4, (vocab, dim))
        h = r.normal(0, .5, (visual+text, dim))
        ids = np.r_[np.arange(visual)*3+11, [-1]*text]
        positions = np.arange(visual+text)*2
        fingerprint = hashlib.sha256(h.tobytes()+ids.tobytes()).hexdigest()
        self.initial = dict(hidden=h, positions=positions, ids=ids, completed=0,
                            kv=[], trace=[], prefix_fingerprint=fingerprint)
        self.targets = (2, 5, 1)

    def capture(self, completed=2):
        state = self.fork(self.initial)
        self.advance(state, completed)
        state["prefix_fingerprint"] = hashlib.sha256(
            state["hidden"].tobytes()+str(completed).encode()).hexdigest()
        return state

    def inspect(self, state):
        return dict(completed=state["completed"],
                    active_ids=tuple(map(int, state["ids"][state["ids"] >= 0])),
                    protected_positions=tuple(map(int, state["positions"][state["ids"] < 0])),
                    visual_positions={int(i): int(p) for i, p in zip(state["ids"], state["positions"]) if i >= 0},
                    kv_support=tuple(tuple(map(int, c["ids"][c["ids"] >= 0])) for c in state["kv"]),
                    prefix_fingerprint=state["prefix_fingerprint"],
                    state_hash=hashlib.sha256(state["hidden"].tobytes()+b"".join(
                        c["k"].tobytes()+c["v"].tobytes() for c in state["kv"])).hexdigest())

    def fork(self, state):
        return deepcopy(state)

    def prune(self, state, keep):
        h, p, v, _ = compact_sequence(state["hidden"], state["positions"], state["ids"], keep)
        state.update(hidden=h, positions=p, ids=v)

    @staticmethod
    def attention(q, k, v, allowed):
        z = q@k.T/np.sqrt(q.shape[-1])
        z = np.where(allowed, z, -np.inf)
        p = np.exp(z-z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return p@v

    def advance(self, state, completed):
        if not state["completed"] <= completed <= self.depth:
            raise ValueError("invalid resume depth")
        for layer in range(state["completed"], completed):
            h, p, ids = state["hidden"], state["positions"], state["ids"]
            q, k, v = (h@w for w in self.weights[layer])
            state["kv"].append(dict(k=k.copy(), v=v.copy(), ids=ids.copy(), positions=p.copy()))
            out = self.attention(q, k, v, p[:, None] >= p[None, :])
            state["hidden"] = h+np.tanh(out)
            state["trace"].append(int((ids >= 0).sum()))
            state["completed"] = layer+1

    def align_cache(self, state, keep):
        for cache in state["kv"]:
            rows = (cache["ids"] < 0) | np.isin(cache["ids"], keep)
            for key in ("k", "v", "ids", "positions"):
                cache[key] = cache[key][rows].copy()

    def _decode(self, state, teacher):
        h = state["hidden"][-1:]
        position = int(state["positions"].max())+1
        losses, generated = [], []
        for i, target in enumerate(self.targets):
            logits = (h@self.output)[0]
            generated.append(int(np.argmax(logits)))
            z = logits-logits.max()
            losses.append(float(np.log(np.exp(z).sum())-z[target]))
            if i == len(self.targets)-1:
                break
            previous = target if teacher else generated[-1]
            h = self.embeddings[previous:previous+1].copy()
            for layer, cache in enumerate(state["kv"]):
                q, k, v = (h@w for w in self.weights[layer])
                cache["k"] = np.concatenate((cache["k"], k))
                cache["v"] = np.concatenate((cache["v"], v))
                cache["ids"] = np.r_[cache["ids"], -1]
                cache["positions"] = np.r_[cache["positions"], position]
                out = self.attention(q, cache["k"], cache["v"],
                                     (cache["positions"] <= position)[None])
                h = h+np.tanh(out)
            position += 1
        return float(np.mean(losses)), generated

    def score_suffix(self, state):
        self.advance(state, self.depth)
        nll, _ = self._decode(self.fork(state), teacher=True)
        _, generated = self._decode(self.fork(state), teacher=False)
        return dict(nll=nll, answer_tokens=len(self.targets),
                    trace=tuple(state["trace"]), backend="numpy_causal_fixture",
                    generated=generated)
````

## `make_catalog.py`

SHA-256: `794b740ccef988182525f5763e8be911f2477f988252f169ec238b6f7c062fee`

````python
"""Generate finite, deduplicated development configs. No VLM is run here."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from elva.budget import Schedule


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_catalog():
    result, seen = [], set()

    def add(block, priority, budget, schedule, guide="pen", cover=True,
            selector="Q_ATTN", beta=.5, final_selector="Q_ATTN", seed=0, conditional=False,
            score_guide="same"):
        assert schedule.work == budget and schedule.depth == 32
        method = dict(budget=budget, original_tokens=576, depth=32,
                      schedule=dict(counts=schedule.counts, lengths=schedule.lengths),
                      stage1=dict(guide=guide, cover=cover, beta=beta if guide == "dual" or score_guide == "dual" else None,
                                  score_guide=score_guide, payload="pen", saliency="BIND_ORIGINAL_SAME_RULE"),
                      stage2=dict(first_selector=selector, later_selector=final_selector,
                                  needs_l9_features=selector == "TRANSIENT_PAIR_PLUS_L9",
                                  seed=seed if "RANDOM" in final_selector else None),
                      merge=False, cache_mode="native", conditional=conditional)
        digest = fingerprint(method)
        if digest in seen:
            return
        seen.add(digest)
        result.append(dict(id=f"ELVA_R2_{len(result):03d}", block=block, priority=priority,
                           config_hash=digest, method=method))

    for budget, counts in [(1024, (60, 28, 12)), (1536, (90, 42, 18)), (2048, (120, 56, 24))]:
        k0, middle, final = counts
        long = Schedule(counts, (10, 10, 12))
        for guide, cover in [("pen", False), ("l9", False), ("dual", False), ("pen", True), ("dual", True)]:
            add("A_stage1", 0 if budget == 1536 else 1, budget, long, guide, cover)
        shapes = [long,
                  Schedule((k0, middle, final), (8, 16, 8)),
                  Schedule((k0, 32*budget//1024, final), (10, 8, 14)),
                  Schedule((k0, 20*budget//1024, final), (12, 8, 12))]
        for shape in shapes:
            for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
                add("B_schedule", 1, budget, shape, selector=selector)
        add("J_stage1_stage2_bridge", 2, budget, long, guide="dual", selector="RDRIFT_NATIVE")
        legacy = {1024: (80, 48, 16), 1536: (96, 64, 32), 2048: (160, 96, 32)}[budget]
        for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
            add("C_legacy_recipe", 1, budget, Schedule(legacy, (3, 10, 19)), selector=selector)
            add("H_stage1_transfer", 2, budget, Schedule(legacy, (3, 10, 19)),
                guide="dual", selector=selector)
        for selector in ("QED_NATIVE", "ATTN_COVER_NATIVE"):
            add("D_selector", 1, budget, long, selector=selector)
        for seed in (11, 23, 47):
            add("E_random_final", 2, budget, long, final_selector="RANDOM_FINAL", seed=seed)
        for beta in (.25, .75):
            add("F_beta", 2, budget, long, guide="dual", beta=beta)
        if budget == 1536:
            add("I_stage1_factor", 2, budget, long, guide="dual", score_guide="pen")
            add("I_stage1_factor", 2, budget, long, guide="pen", score_guide="dual")
        # Conditional heads use forced-final identity F, including their own
        # matched anchor. They CANNOT be compared only to natural re-selection.
        for selector in ("FROZEN_FINAL_ANCHOR", "TRANSIENT_PAIR_CURRENT", "TRANSIENT_PAIR_PLUS_L9"):
            add("G_conditional", 3, budget, long, selector=selector, conditional=True)
    return sorted(result, key=lambda c: (c["priority"], c["id"]))


def main():
    root = Path(__file__).resolve().parent
    out = root/"configs"
    out.mkdir(exist_ok=True)
    catalog = build_catalog()
    for config in catalog:
        (out/f"{config['id']}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n")
    summary = dict(configs=len(catalog), blocks=dict(Counter(c["block"] for c in catalog)),
                   unconditional=sum(not c["method"]["conditional"] for c in catalog),
                   conditional=sum(c["method"]["conditional"] for c in catalog),
                   four_task_jobs=4*len(catalog),
                   note="development capacity, not completed experiments; native source aliases must be deduplicated")
    (out/"catalog.json").write_text(json.dumps(dict(summary=summary, configs=catalog), indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

````

## `run_queue.py`

SHA-256: `d5ee86916388976bb5ebf40dd6ccbdceb05d7382037d98bddeaa36079cdeaa18`

````python
"""Resumable three-worker ACC queue; caller supplies the REAL native runner.

No shell interpolation. GPU IDs are configurable; reserve a separate card
for TTFT. Official scorer output and complete trace are mandatory for DONE.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import threading
import time


REQUIRED_BINDING = ("source_commit", "model_sha", "processor_sha", "scorer_sha",
                    "prompt_sha", "tokenizer_sha", "dataset_sha", "backend", "cache_mode",
                    "native_selectors", "original_tokens", "depth")


def run_process(argv, *, env, log, timeout):
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def conditional_ready(config, binding):
    if not config["method"]["conditional"]:
        return True
    selector = config["method"]["stage2"]["first_selector"]
    if not binding.get("transient_adapter_ready", False):
        return False
    if selector == "FROZEN_FINAL_ANCHOR":
        return True
    gate = binding.get("transient_gate", {})
    return (gate.get("schema") == "transient_survival_swap_v1"
            and gate.get("status") == "READY" and gate.get("cache_mode") == "native"
            and bool(gate.get("weights_sha")) and bool(gate.get("catalog_hash"))
            and gate.get("real_labels") is True and gate.get("heldout_gain_over_noise") is True
            and gate.get("generation_checked") is True)


def validate_result(value, config, binding, task, run_key):
    method = config["method"]
    if value.get("run_key") != run_key or value.get("config_hash") != config["config_hash"]:
        raise ValueError("stale or missing provenance fingerprint")
    if value.get("task") != task or value.get("backend") != binding["backend"] or value.get("cache_mode") != method["cache_mode"]:
        raise ValueError("task/backend/cache mismatch")
    count = value.get("samples")
    if type(count) is not int or count <= 0 or value.get("scorer_sha") != binding["scorer_sha"]:
        raise ValueError("real scored samples and bound scorer required")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics or not all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()):
        raise ValueError("missing/nonfinite official metrics")
    expected = [k for k, d in zip(method["schedule"]["counts"], method["schedule"]["lengths"]) for _ in range(d)]
    audit = value.get("trace_audit", {})
    if (audit.get("expected") != expected or audit.get("observed_samples") != count
            or audit.get("mismatch_count") != 0):
        raise ValueError("missing/incorrect all-sample trace audit")
    if value.get("binding_hash") != digest(binding):
        raise ValueError("binding changed; do not reuse old results")


def run(catalog, binding, command, output, gpus, tasks, hours, allow_conditional=False):
    if any(key not in binding or not binding[key] for key in REQUIRED_BINDING):
        raise ValueError("complete server binding manifest required")
    if (binding["original_tokens"], binding["depth"]) != (576, 32):
        raise ValueError("this catalog is for 576x32; regenerate budgets for other architectures")
    if not gpus or len(set(gpus)) != len(gpus) or hours <= 0 or not math.isfinite(hours):
        raise ValueError("distinct ACC GPU IDs and finite positive time budget required")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("runner command must be an argv array")
    required = ("{config}", "{task}", "{out}", "{run_key}", "{binding}")
    if any(not any(p in token for token in command) for p in required):
        raise ValueError("runner argv missing mandatory routing placeholders")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/"binding.json").write_text(json.dumps(binding, indent=2))
    deadline = time.monotonic()+hours*3600
    jobs, states, lock = [], [], threading.Lock()
    for config in catalog:
        if config["method"]["conditional"] and (not allow_conditional or not conditional_ready(config, binding)):
            states.append(dict(config=config["id"], status="DEFERRED_CONDITIONAL"))
            continue
        for task in tasks:
            key = digest(dict(config=config, binding=binding, task=task))
            jobs.append((config, task, key))

    def worker(gpu):
        while time.monotonic() < deadline:
            with lock:
                if not jobs:
                    return
                config, task, key = jobs.pop(0)
            path = out/key
            path.mkdir(exist_ok=True)
            result_path, config_path = path/"result.json", path/"config.json"
            config_path.write_text(json.dumps(config, indent=2))
            status = dict(config=config["id"], task=task, run_key=key, gpu=gpu)
            try:
                if result_path.exists():
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "REUSED_VERIFIED"
                else:
                    replacements = {"{config}": str(config_path), "{task}": task,
                                    "{out}": str(result_path), "{run_key}": key,
                                    "{binding}": str(out/"binding.json")}
                    argv = command.copy()
                    for marker, value in replacements.items():
                        argv = [token.replace(marker, value) for token in argv]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    remaining = max(.01, deadline-time.monotonic())
                    with (path/"runner.log").open("w") as log:
                        run_process(argv, env=env, log=log, timeout=remaining)
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "DONE"
            except Exception as error:
                status.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
                if result_path.exists():
                    result_path.replace(path/f"rejected-{time.time_ns()}.json")
            with lock:
                states.append(status)
                (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    states.extend(dict(config=c["id"], task=t, run_key=k, status="PENDING_DEADLINE") for c, t, k in jobs)
    (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))
    return states


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="configs/catalog.json")
    p.add_argument("--binding", required=True)
    p.add_argument("--runner", required=True, help="JSON file containing argv array")
    p.add_argument("--out", default="results")
    p.add_argument("--gpus", nargs="+", default=["0", "2", "4"])
    p.add_argument("--tasks", nargs="+", default=["GQA", "POPE", "MME", "MMBench"])
    p.add_argument("--hours", type=float, default=6.)
    p.add_argument("--allow-conditional", action="store_true")
    args = p.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    run(load(args.catalog)["configs"], load(args.binding), load(args.runner), args.out,
        args.gpus, args.tasks, args.hours, args.allow_conditional)
````

## `tests/test_elva.py`

SHA-256: `1b242201f6b9837f33f8dd10caae9e8d24f6804e683a51ce92de46f6d400ad53`

````python
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from elva.budget import Schedule, Trace, feasible_suffixes, choose_suffix
from elva.evidence import (unit, select_stage1, question_visual_mass, rank_drift,
                           average_ranks, compact_sequence)
from elva.counterfactual import Swap, run_swap, answer_nll, ESTIMAND
from elva.toy_adapter import ToyAdapter
from elva.learning import (PairModel, pair_features, split_image, fit_pairs,
                           choose_swap, calibrate_threshold)
from elva.integration import BoundRuntime
from elva.capture import LayerCapture
from make_catalog import build_catalog
from run_queue import digest, validate_result, run_process, conditional_ready, run as run_queue
from elva.benchmark import paired_ttft, summarize
from elva.policy import propose, feature_catalog, decide, collect_catalog


class BudgetTests(unittest.TestCase):
    def test_requested_plateau_and_inverse(self):
        for counts, budget in [((60, 28, 12), 1024), ((90, 42, 18), 1536), ((120, 56, 24), 2048)]:
            p = Schedule(counts, (10, 10, 12))
            self.assertEqual(p.work, budget)
            self.assertEqual(p.events, ((10, counts[1]), (20, counts[2])))
            self.assertEqual(sum(p.trace), budget)
            self.assertEqual((1-p.compression(576))*576*32, budget)
            self.assertEqual(p.trace[9], counts[0])
            self.assertEqual(p.trace[10], counts[1])
            self.assertEqual(p.trace[20], counts[2])

    def test_naive_request_is_over_budget(self):
        self.assertEqual(Schedule((60, 30, 16), (10, 11, 11)).work, 1106)
        self.assertNotEqual(1106, 1024)

    def test_invalid_schedules(self):
        for counts, lengths in [((10, 10), (2, 2)), ((10, 20), (2, 2)), ((10,), (0,)),
                                ((0,), (2,)), ((True,), (2,)), ((10,), (2.5,)), ((), ())]:
            with self.subTest(counts=counts, lengths=lengths), self.assertRaises(ValueError):
                Schedule(counts, lengths)

    def test_square_cost_is_not_work(self):
        a, b = Schedule((60, 28, 12), (10, 10, 12)), Schedule((60, 32, 12), (10, 8, 14))
        self.assertEqual(a.work, b.work)
        self.assertNotEqual(a.proxies(7), b.proxies(7))

    def test_trace_records_before_block(self):
        p = Schedule((5, 2), (2, 3))
        trace = Trace(p.work, p.depth)
        for block, k in enumerate(p.trace, 1):
            trace.before_block(block, k)
        p.validate_trace(trace.finish())
        with self.assertRaises(ValueError):
            trace.before_block(5, 2)

    def test_trace_rejects_resurrection_and_shortfall(self):
        trace = Trace(20, 4)
        trace.before_block(1, 4)
        with self.assertRaises(ValueError):
            trace.before_block(2, 5)
        with self.assertRaises(ValueError):
            trace.finish()

    def test_suffix_solver_against_exhaustive_integer_enumeration(self):
        # Independent brute-force baseline over ALL small catalogs.
        cases = 0
        for depth in range(3, 8):
            for completed in range(1, depth-1):
                boundaries = tuple(range(completed+1, depth))
                for current in range(2, 7):
                    spent = completed*current
                    for budget in range(spent+1, spent+(depth-completed)*current+1):
                        finals = tuple(range(1, current))
                        actual = feasible_suffixes(depth=depth, completed=completed, spent=spent,
                                                  current_k=current, budget=budget,
                                                  boundaries=boundaries, finals=finals)
                        actual_set = {(p.counts, p.lengths) for p in actual}
                        brute = set()
                        for boundary, middle, final in itertools.product(boundaries, range(1, current+1), finals):
                            d1, d2 = boundary-completed, depth-boundary
                            if final < middle and spent+middle*d1+final*d2 == budget:
                                brute.add(((middle, final), (d1, d2)))
                        self.assertEqual(actual_set, brute)
                        cases += 1
        self.assertEqual(cases, 1000)

    def test_suffix_choice_abstains_and_rejects_cost_mismatch(self):
        catalog = feasible_suffixes(depth=32, completed=10, spent=600, current_k=60,
                                    budget=1024, boundaries=(18, 20, 26), finals=(12,))
        self.assertEqual(len(catalog), 3)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .02], threshold=.05), 0)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .2], threshold=.05), 2)
        with self.assertRaises(ValueError):
            choose_suffix([catalog[0], Schedule((5,), (2,))], [0, 1])
        with self.assertRaises(ValueError):
            choose_suffix(catalog, [0, float("nan"), 2])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(3)
        self.pen, self.early = r.normal(size=(8, 5)), r.normal(size=(8, 5))
        self.ids = np.arange(8)*7+100
        self.a, self.b = np.arange(8)/40, np.arange(8)[::-1]/40

    def select(self, **kw):
        return select_stage1(self.pen, self.a, self.ids, 3, early=self.early,
                             early_scores=self.b, early_ids=self.ids, **kw)

    def test_payload_and_projector_width_protected(self):
        for guide in ("pen", "l9", "dual"):
            out = self.select(guide=guide)
            np.testing.assert_array_equal(out["payload"], self.pen[out["keep"]])
            self.assertEqual(out["payload"].shape, (3, 5))

    def test_dual_kernel_is_weighted_cosine(self):
        beta = .3
        out = self.select(guide="dual", beta=beta)
        p = self.pen/np.linalg.norm(self.pen, axis=1, keepdims=True)
        e = self.early/np.linalg.norm(self.early, axis=1, keepdims=True)
        np.testing.assert_allclose(out["metric"]@out["metric"].T,
                                   (1-beta)*(p@p.T)+beta*(e@e.T), atol=1e-12)

    def test_dual_endpoints_match_single_layer(self):
        for beta, single in [(0., "pen"), (1., "l9")]:
            dual, one = self.select(guide="dual", beta=beta), self.select(guide=single)
            np.testing.assert_array_equal(dual["keep"], one["keep"])
            np.testing.assert_array_equal(dual["metric"], one["metric"])

    def test_geometry_and_score_factor_ablation(self):
        geometry_only = self.select(guide="dual", score_guide="pen")
        np.testing.assert_array_equal(geometry_only["scores"], self.a)
        score_only = self.select(guide="pen", score_guide="dual")
        np.testing.assert_array_equal(score_only["metric"], self.pen)
        np.testing.assert_array_equal(score_only["scores"], (self.a+self.b)/2)

    def test_original_cover_required(self):
        with self.assertRaises(RuntimeError):
            self.select(use_cover=True)

    def test_mutating_callback_cannot_change_original_payload(self):
        before = self.pen.copy()
        def malicious(**kwargs):
            kwargs["metric_features"][:] = 0
            kwargs["scores"][:] = 0
            kwargs["patch_ids"][:] = 999
            return np.array([0, 3, 6])
        out = self.select(use_cover=True, cover=malicious)
        np.testing.assert_array_equal(self.pen, before)
        np.testing.assert_array_equal(out["payload"], before[[0, 3, 6]])

    def test_zero_vector_finite_and_frozen_ids(self):
        self.pen[0] = 0
        out = self.select(guide="dual", frozen_keep=[0, 4, 7])
        self.assertTrue(np.isfinite(out["metric"]).all())
        np.testing.assert_array_equal(out["ids"], self.ids[[0, 4, 7]])

    def test_tiny_normalization_matches_documented_epsilon(self):
        result = unit([[0., 0.], [1e-30, 0.]])
        np.testing.assert_allclose(result, [[0, 0], [1e-18, 0]], atol=0)

    def test_alignment_nonfinite_and_duplicate_keep_rejected(self):
        with self.assertRaises(ValueError):
            select_stage1(self.pen, self.a, self.ids, 3, guide="l9", early=self.early,
                          early_scores=self.b, early_ids=self.ids[::-1])
        with self.assertRaises(ValueError):
            self.select(frozen_keep=[0, 0, 2])
        self.pen[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.select()

    def test_causal_normalization_all_visible_keys(self):
        z = np.zeros((2, 4))
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], bool)
        mass = question_visual_mass(z, mask, [0, 1])
        np.testing.assert_allclose(mass, [.375, .375])
        self.assertLess(mass.sum(), 1)

    def test_attention_mask_padding_extremes(self):
        z = np.array([[10000, -10000, np.nan]])
        np.testing.assert_allclose(question_visual_mass(z, np.array([[1, 1, 0]], bool), [0]), [1])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.zeros_like(z, dtype=bool), [0])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.array([[1, 1, 0]]), [0])

    def test_rank_ties_and_history_original_id_alignment(self):
        np.testing.assert_allclose(average_ranks([5, 5, 5]), [.5, .5, .5])
        a = rank_drift([3, 1, 2], [20, 50, 10], [1, 2, 99, 3], [50, 10, 99, 20])
        b = rank_drift([3, 1, 2], [20, 50, 10], [3, 1, 2], [20, 50, 10])
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            rank_drift([3, 1], [20, 50], [1], [20])

    def test_compaction_keeps_text_and_original_positions(self):
        h = np.arange(24).reshape(6, 4).astype(float)
        compact, p, ids, rows = compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [4])
        np.testing.assert_array_equal(rows, [0, 2, 4, 5])
        np.testing.assert_array_equal(p, [2, 7, 12, 20])
        np.testing.assert_array_equal(compact, h[rows])
        with self.assertRaises(ValueError):
            compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [999])


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ToyAdapter()
        self.prefix = self.adapter.capture(2)
        self.swap = Swap(2, 5, (11, 14, 17), (20, 23), (26, 29), (11, 14))

    def execute(self, mode="native", swap=None):
        return run_swap(self.adapter, self.prefix, swap or self.swap,
                        image_key="fixture-image", question_key="fixture-question", mode=mode)

    def test_actual_causal_decoder_branches_and_work(self):
        out = self.execute()
        self.assertEqual(out["trace"], [12, 12, 5, 5, 5, 2, 2, 2])
        self.assertEqual(out["work"], 45)
        self.assertTrue(math.isfinite(out["delta"]))
        self.assertEqual(out["final_ids"], [11, 14])

    def test_swap_sign_reverses(self):
        reverse = Swap(2, 5, self.swap.common, self.swap.group_b, self.swap.group_a, self.swap.final)
        self.assertAlmostEqual(self.execute()["delta"], -self.execute(swap=reverse)["delta"], places=12)

    def test_support_alignment_changes_cache_estimuand(self):
        native, aligned = self.execute(), self.execute("aligned_support")
        self.assertNotAlmostEqual(native["nll_a"], aligned["nll_a"], places=10)
        self.assertEqual(aligned["cache_mode"], "aligned_support")

    def test_repeated_noop_actual_forwards_are_identical(self):
        values = []
        for _ in range(4):
            state = self.adapter.fork(self.prefix)
            self.adapter.prune(state, self.swap.common+self.swap.group_a)
            self.adapter.advance(state, 5)
            self.adapter.prune(state, self.swap.final)
            values.append(self.adapter.score_suffix(state)["nll"])
        self.assertEqual(max(values)-min(values), 0)

    def test_shared_prefix_immutable(self):
        before = self.adapter.inspect(self.prefix)
        self.execute()
        self.assertEqual(before, self.adapter.inspect(self.prefix))

    def test_aliasing_fork_is_rejected(self):
        self.adapter.fork = lambda state: state
        with self.assertRaises(ValueError):
            self.execute()

    def test_invalid_sets_and_unknown_group(self):
        for args in [(2, 2, (1,), (2,), (3,), (1,)),
                     (2, 5, (1,), (1,), (3,), (1,)),
                     (2, 5, (1,), (2,), (3, 4), (1,)),
                     (2, 5, (1,), (2,), (3,), (7,))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Swap(*args)
        unknown = Swap(2, 5, (11, 14), (999,), (20,), (11,))
        with self.assertRaises(ValueError):
            self.execute(swap=unknown)

    def test_answer_nll_shift_not_same_position(self):
        logits = np.log([[.8, .2], [.6, .4], [.1, .9]])
        actual = answer_nll(logits, [0, 1, 0], np.array([0, 1, 1], bool))
        self.assertAlmostEqual(actual, (-math.log(.2)-math.log(.6))/2)
        with self.assertRaises(ValueError):
            answer_nll(logits, [0, 1, 0], np.zeros(3, bool))


class LearningTests(unittest.TestCase):
    def test_shared_policy_catalog_no_resurrection_and_limit(self):
        ids = np.arange(60)*3+19
        scores = np.arange(60)/60
        catalog = propose(scores, ids, np.arange(28), 12)
        self.assertEqual(len(catalog.swaps), 15)
        self.assertTrue(set(catalog.final).issubset(catalog.baseline))
        for common, inserted, removed in catalog.swaps:
            self.assertEqual(len(common)+len(inserted), 28)
            self.assertEqual(len(inserted), len(removed))
            self.assertTrue(set(catalog.final).issubset(common))
        features = np.c_[scores, scores**2]
        matrix = feature_catalog(catalog, features, begin=10, end=20, depth=32)
        model = PairModel(np.ones(8), np.zeros(8), "native", tuple("abcdefgh"), catalog.rule_hash)
        self.assertEqual(decide(catalog, features, model, begin=10, end=20, depth=32, threshold=0)["action"], 0)
        with self.assertRaises(ValueError):
            feature_catalog(catalog, features, begin=10, end=21, depth=32)

    def test_policy_collects_real_fixture_forwards_and_blocks_training(self):
        adapter = ToyAdapter()
        prefix = adapter.capture(2)
        ids = np.array(adapter.inspect(prefix)["active_ids"])
        scores = np.arange(12)/12
        catalog = propose(scores, ids, np.arange(7), 2, group_size=2, limit=4, begin=2, end=5, depth=8)
        x = scores[:, None]
        rows = collect_catalog(adapter, prefix, catalog, ids, x, begin=2, end=5, depth=8,
                               image_key="fixture", question_key="question", feature_names=list("abcd"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["work"] for row in rows}, {51})
        self.assertTrue(all(row["fixture"] for row in rows))
        with self.assertRaises(ValueError):
            fit_pairs(rows, feature_names=list("abcd"), catalog_hash=catalog.rule_hash, cache_mode="native")

    def records(self):
        keys = [f"test-generated-{i}" for i in range(100) if split_image(f"test-generated-{i}") == "train"]
        result = []
        for i in range(20):
            x = np.array([i/20, (i % 3)-1.])
            result.append(dict(schema=ESTIMAND, cache_mode="native", catalog_hash="catalog",
                               feature_names=["a", "b"], feature_cutoff=10, begin=10,
                               image_key=keys[i % 5], x=x.tolist(), delta=float(x@np.array([2., -.3]))))
        return result

    def test_pair_features_context_and_antisymmetry(self):
        x = np.arange(30).reshape(10, 3)/10
        a = pair_features(x, [0, 1], [2, 3], [4, 5], begin=10, end=20, depth=32)
        b = pair_features(x, [0, 1], [4, 5], [2, 3], begin=10, end=20, depth=32)
        np.testing.assert_array_equal(a, -b)
        changed = pair_features(x, [6, 7], [2, 3], [4, 5], begin=10, end=20, depth=32)
        self.assertFalse(np.allclose(a, changed))

    def test_grouped_split_stable(self):
        self.assertEqual(split_image("canonical-image"), split_image("canonical-image"))
        self.assertEqual(set(split_image(f"img-{i}") for i in range(100)), {"train", "validation", "test"})

    def test_fit_save_load_and_prediction(self):
        records = self.records()
        model = fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native", ridge=1e-6)
        x = np.array([r["x"] for r in records])
        np.testing.assert_allclose(model.predict(x), [r["delta"] for r in records], atol=1e-6)
        np.testing.assert_allclose(model.predict(-x), -model.predict(x))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"head.json"
            model.save(path)
            loaded = PairModel.load(path, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")
            np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
            with self.assertRaises(ValueError):
                PairModel.load(path, feature_names=["a", "b"], catalog_hash="old", cache_mode="native")

    def test_old_schema_future_features_and_fixture_rejected(self):
        for key, value in [("schema", "old_e020"), ("feature_cutoff", 11), ("fixture", True), ("cache_mode", "aligned_support")]:
            records = self.records()
            records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_validation_images_cannot_enter_fit(self):
        records = self.records()
        records[0]["image_key"] = next(f"held-{i}" for i in range(100) if split_image(f"held-{i}") != "train")
        with self.assertRaises(ValueError):
            fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_candidate_max_calibration_and_all_noop(self):
        p = np.array([[0, .8, .2], [0, .1, .6]])
        good = calibrate_threshold(p, [[0, .4, -.3], [0, -.1, .2]], [0, 0, 0])
        bad = calibrate_threshold(p, [[0, -.4, -.3], [0, -.1, -.2]], [0, 0, 0])
        self.assertEqual(good["status"], "CANDIDATE")
        self.assertEqual(bad["status"], "NO_SUPPORTED_DECISIONS")
        self.assertEqual(bad["accepted"], 0)
        self.assertEqual(choose_swap([0, .1], threshold=.2), 0)


class IntegrationAndCatalogTests(unittest.TestCase):
    def test_queue_runs_resumes_and_recovers_bad_result(self):
        # Mock runner lives only in a temporary directory and labels its
        # metrics/binding as fixtures. No model score is produced.
        source = '''import sys,json,hashlib,pathlib
cfg,task,out,key,binding=sys.argv[1:]
c=json.loads(pathlib.Path(cfg).read_text()); b=json.loads(pathlib.Path(binding).read_text())
m=c['method']; s=m['schedule']; trace=[k for k,d in zip(s['counts'],s['lengths']) for _ in range(d)]
h=lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
value=dict(run_key=key,config_hash=c['config_hash'],task=task,backend=b['backend'],cache_mode=m['cache_mode'],samples=2,scorer_sha=b['scorer_sha'],metrics={'fixture_metric':1.},binding_hash=h(b),trace_audit=dict(expected=trace,observed_samples=2,mismatch_count=0))
pathlib.Path(out).write_text(json.dumps(value))
'''
        configs = build_catalog()[:2]
        binding = dict(source_commit="FIXTURE", model_sha="FIXTURE", processor_sha="FIXTURE",
                       scorer_sha="FIXTURE", prompt_sha="FIXTURE", tokenizer_sha="FIXTURE",
                       dataset_sha="FIXTURE", backend="FIXTURE", cache_mode="native",
                       native_selectors={"fixture": "FIXTURE"}, original_tokens=576, depth=32)
        with tempfile.TemporaryDirectory() as d:
            runner, output = Path(d)/"runner.py", Path(d)/"out"
            runner.write_text(source)
            command = [sys.executable, str(runner), "{config}", "{task}", "{out}", "{run_key}", "{binding}"]
            args = (configs, binding, command, output, ["0", "1"], ["FIXTURE_TASK"], .01)
            self.assertEqual([s["status"] for s in run_queue(*args)], ["DONE", "DONE"])
            self.assertEqual([s["status"] for s in run_queue(*args)], ["REUSED_VERIFIED", "REUSED_VERIFIED"])
            path = next(output.glob("*/result.json"))
            value = json.loads(path.read_text())
            value["trace_audit"]["mismatch_count"] = 1
            path.write_text(json.dumps(value))
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["FAILED", "REUSED_VERIFIED"])
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["DONE", "REUSED_VERIFIED"])

    def test_conditional_queue_requires_new_real_evidence(self):
        config = next(c for c in build_catalog() if c["method"]["stage2"]["first_selector"] == "TRANSIENT_PAIR_CURRENT")
        self.assertFalse(conditional_ready(config, {}))
        binding = dict(transient_adapter_ready=True, transient_gate=dict(
            schema=ESTIMAND, status="READY", cache_mode="native", weights_sha="test-only",
            catalog_hash="test-only", real_labels=True, heldout_gain_over_noise=True, generation_checked=True))
        self.assertTrue(conditional_ready(config, binding))
        binding["transient_gate"]["schema"] = "old_e020"
        self.assertFalse(conditional_ready(config, binding))

    def test_first_token_callback_timing_and_missing_token(self):
        rows = paired_ttft(["fixture0", "fixture1"], ("a", "b"), lambda s, m: s+m,
                           lambda: None, rounds=2, warmup=0)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summarize(rows)["a"]["observations"], 4)
        with self.assertRaises(ValueError):
            paired_ttft([0], ("a", "b"), lambda s, m: None, lambda: None, warmup=0)

    def test_capture_completed_block_indices_and_removal(self):
        class Handle:
            def __init__(self, hooks, fn): self.hooks, self.fn = hooks, fn
            def remove(self): self.hooks.remove(self.fn)
        class Block:
            def __init__(self, i): self.i, self.hooks = i, []
            def register_forward_hook(self, fn):
                self.hooks.append(fn)
                return Handle(self.hooks, fn)
            def __call__(self, x):
                y = x+self.i
                for fn in self.hooks: fn(self, (x,), (y,))
                return y
        blocks = [Block(i) for i in range(1, 13)]
        hidden = [np.zeros((1, 5, 3))]
        with LayerCapture(blocks) as capture:
            for block in blocks: hidden.append(block(hidden[-1]))
            early, pen = capture.get()
            np.testing.assert_array_equal(early, hidden[9])
            np.testing.assert_array_equal(pen, hidden[-2])
        self.assertTrue(all(not block.hooks for block in blocks))

    def test_capture_invalid_layer_rejected(self):
        with self.assertRaises(ValueError):
            LayerCapture([None]*8, early_completed=9)

    def test_catalog_distinct_and_all_budgets_exact(self):
        configs = build_catalog()
        self.assertEqual(len(configs), len({c["config_hash"] for c in configs}))
        self.assertEqual(len(configs), 83)
        self.assertEqual(sum(not c["method"]["conditional"] for c in configs), 74)
        for c in configs:
            p = Schedule(**c["method"]["schedule"])
            self.assertEqual(p.work, c["method"]["budget"])
            self.assertEqual(p.depth, 32)

    def test_native_runtime_callbacks_and_positions(self):
        state = dict(ids=[101, 303, 202, 404, 505], protected=[1, 10])
        def inspect(s):
            return dict(active_ids=s["ids"], protected_positions=s["protected"], backend="bound")
        def select(s, *, k, **kwargs):
            return s["ids"][:k]
        def compact(s, *, keep_ids):
            return dict(ids=list(keep_ids), protected=s["protected"].copy())
        p = Schedule((5, 2), (2, 3))
        rt = BoundRuntime(p, select, compact, inspect)
        for block in range(1, 6):
            rt.before_block(block, state)
            state = rt.after_block(block, state, selector="SOURCE_BOUND")
        self.assertEqual(rt.finish(), p.trace)

    def test_runtime_rejects_backend_change(self):
        p = Schedule((2, 1), (1, 1))
        state = dict(ids=[1, 2], backend="flash")
        inspect = lambda s: dict(active_ids=s["ids"], protected_positions=[7], backend=s["backend"])
        rt = BoundRuntime(p, lambda *a, **k: [1], lambda *a, **k: dict(ids=[1], backend="eager"), inspect)
        rt.before_block(1, state)
        with self.assertRaises(ValueError):
            rt.after_block(1, state, selector="native")

    def test_result_reuse_requires_trace_and_provenance(self):
        config = build_catalog()[0]
        binding = dict(backend="fixture", scorer_sha="fixture")
        schedule = Schedule(**config["method"]["schedule"])
        value = dict(run_key="key", config_hash=config["config_hash"], task="GQA", backend="fixture",
                     cache_mode="native", samples=2, scorer_sha="fixture", metrics={"fixture_only": 1.},
                     binding_hash=digest(binding), trace_audit=dict(expected=list(schedule.trace), observed_samples=2, mismatch_count=0))
        validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 1
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 0
        value["metrics"] = {"fixture_only": float("nan")}
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")

    def test_timeout_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/"should-not-exist"
            child = "import time,pathlib; time.sleep(1); pathlib.Path("+repr(str(marker))+").touch()"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(9)"
            with (Path(d)/"log").open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, "-c", parent], env=None, log=log, timeout=.15)
            # Parent and child receive group termination; wait through the
            # child's hypothetical write deadline using a short test subprocess.
            subprocess.run([sys.executable, "-c", "import time; time.sleep(1.1)"], check=True)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `tests/test_torch_parity.py`

SHA-256: `b0378a536b44edbdc63dfa47f41ba12993f646a8090612f675f4e5f3f7bd171d`

````python
"""Run on server before ACC; absence of PyTorch is an explicit skip."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is absent in this workspace")
class TorchParityTests(unittest.TestCase):
    def test_numpy_parity_and_payload_protection(self):
        import torch
        from elva.evidence import select_stage1 as ref, question_visual_mass as mass_ref
        from elva.torch_ops import select_stage1, question_visual_mass
        r = np.random.default_rng(4)
        x, e = r.normal(size=(12, 6)), r.normal(size=(12, 6))
        a, b, ids = r.random(12), r.random(12), np.arange(12)*3
        for guide, beta in [("pen", .5), ("l9", .5), ("dual", 0), ("dual", .5), ("dual", 1)]:
            options = dict(guide=guide, beta=beta)
            reference = ref(x, a, ids, 4, early=e, early_scores=b, early_ids=ids, **options)
            out = select_stage1(torch.tensor(x), torch.tensor(a), torch.tensor(ids), 4,
                                early=torch.tensor(e), early_scores=torch.tensor(b), early_ids=torch.tensor(ids), **options)
            np.testing.assert_array_equal(out["keep"].numpy(), reference["keep"])
            np.testing.assert_array_equal(out["payload"].numpy(), reference["payload"])
            np.testing.assert_allclose(out["metric"].numpy(), reference["metric"], atol=1e-6)
        z = r.normal(size=(2, 3, 15))
        mask = np.ones_like(z, bool)
        cols = np.array([1, 3, 7])
        actual = question_visual_mass(torch.tensor(z), torch.tensor(mask), torch.tensor(cols)).numpy()
        np.testing.assert_allclose(actual, mass_ref(z, mask, cols), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
````

## `verify.py`

SHA-256: `0a21d1ef87a945a596725bf3240fb299fe49832ee9b914cc2ffa532ad8781605`

````python
"""Reproducible local checks. All decoder outputs here are FIXTURES."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import compileall
import io
import json
import platform
import unittest
import numpy as np
from elva.budget import Schedule, feasible_suffixes
from elva.toy_adapter import ToyAdapter
from elva.policy import propose, collect_catalog
from make_catalog import build_catalog, main as generate


def main():
    root = Path(__file__).resolve().parent
    out = root/"validation"
    out.mkdir(exist_ok=True)
    generate()
    assert compileall.compile_dir(str(root/"elva"), quiet=1)
    assert compileall.compile_file(str(root/"run_queue.py"), quiet=1)
    catalog = build_catalog()
    budgets = []
    for c in catalog:
        p = Schedule(**c["method"]["schedule"])
        w = c["method"]["budget"]
        assert sum(p.trace) == p.work == w
        assert (1-p.compression(576))*576*p.depth == w
        budgets.append(dict(id=c["id"], counts=p.counts, lengths=p.lengths,
                            trace=p.trace, events=p.events, work=w,
                            compression_exact=str(p.compression(576)), **p.proxies(0)))
    (out/"budgets.json").write_text(json.dumps(budgets, indent=2))
    suffix = feasible_suffixes(depth=32, completed=10, spent=600, current_k=60,
                               budget=1024, boundaries=(18, 20, 26), finals=(12,))
    (out/"feasible_suffixes.json").write_text(json.dumps([
        dict(counts=p.counts, lengths=p.lengths, remaining_work=p.work,
             total_work=600+p.work) for p in suffix], indent=2))
    adapter = ToyAdapter()
    prefix = adapter.capture(2)
    ids = np.array(adapter.inspect(prefix)["active_ids"])
    scores = np.arange(len(ids))/len(ids)
    action_catalog = propose(scores, ids, np.arange(7), 2, group_size=2, limit=4,
                              begin=2, end=5, depth=8, proposal_rule="FIXTURE_ONLY")
    rows = []
    for mode in ("native", "aligned_support"):
        rows.extend(collect_catalog(adapter, prefix, action_catalog, ids, scores[:, None],
                                    begin=2, end=5, depth=8, image_key="FIXTURE_IMAGE",
                                    question_key="FIXTURE_QUESTION", feature_names=list("abcd"), cache_mode=mode))
    assert all(r["fixture"] for r in rows)
    (out/"toy_swaps_FIXTURE_ONLY.json").write_text(json.dumps(rows, indent=2))
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root/"tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out/"tests.txt").write_text(stream.getvalue())
    summary = dict(python=platform.python_version(), tests=result.testsRun,
                   passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                   skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                   failures=len(result.failures), errors=len(result.errors),
                   exhaustive_suffix_catalog_cases=1000,
                   config_count=len(catalog), blocks=dict(Counter(c["block"] for c in catalog)),
                   unconditional=sum(not c["method"]["conditional"] for c in catalog),
                   conditional=sum(c["method"]["conditional"] for c in catalog),
                   four_task_capacity=4*len(catalog), formal_two_methods_three_budgets_nine_tasks=2*3*9,
                   dense_nine_tasks_separate=9,
                   old_shift_work=172*3+116*10+32*19,
                   old_shift_compression_percent=float(100*(1-Fraction(2284, 576*32))),
                   pretrained_vlm_accuracy_run=False, gpu_ttft_run=False,
                   torch_path_runtime_verified=False,
                   note="CPU reference, finite catalog and random causal decoder only; no pretrained-model claims")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(stream.getvalue())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
````

