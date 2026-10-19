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
