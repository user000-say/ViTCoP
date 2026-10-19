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

