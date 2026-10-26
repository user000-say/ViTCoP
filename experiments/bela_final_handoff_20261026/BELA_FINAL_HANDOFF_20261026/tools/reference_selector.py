"""CPU reference for the one conditional candidate in this handoff.

No VLM, labels, training, feature merging, or GPU timing is implemented here.
Port the exact semantics to the production tensor path after the pilot gate.
Original IDs index the full, already encoded visual grid. Score vectors are
full-grid arrays; unavailable current-query scores must not be synthesized.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


def _vector(x, n=None):
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 1 or (n is not None and len(out) != n):
        raise ValueError("expected a full-grid score vector")
    if not np.isfinite(out).all():
        raise ValueError("non-finite score")
    return out


def _ids(values, n):
    out = [int(x) for x in values]
    if len(out) != len(set(out)) or any(x < 0 or x >= n for x in out):
        raise ValueError("IDs must be unique original visual-patch IDs")
    return out


def _affinity(x, n):
    out = np.asarray(x, dtype=np.float64)
    if out.shape != (n, n) or not np.isfinite(out).all():
        raise ValueError("expected finite full-grid affinity")
    if (out < -1e-8).any() or (out > 1 + 1e-8).any():
        raise ValueError("affinity must be clipped to [0,1]")
    return np.clip(out, 0.0, 1.0)


def rank01(scores):
    """Average-rank percentile: ties remain ties; larger input means larger rank."""
    x = _vector(scores)
    n = len(x)
    if n == 0:
        return x.copy()
    if n == 1:
        return np.full(1, 0.5)
    unique, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ranks = (starts + (counts - 1) / 2.0) / (n - 1)
    return ranks[inv]


def disagreement(guide9, guide_late):
    a = _vector(guide9)
    b = _vector(guide_late, len(a))
    return np.maximum(rank01(a) - rank01(b), 0.0)


def bounded_admission(base_ids, guide9, guide_late, entry_priority,
                      affinity9, max_replace):
    """Replace at most b IDs, without changing input features or entry K.

    Native entry priority defines the donors. The strongest existing b positive
    disagreement candidates are excluded from the donor list. Random-admission
    controls must reuse the returned donors and the actual replacement count.
    """
    g9 = _vector(guide9)
    n = len(g9)
    late = _vector(guide_late, n)
    priority = _vector(entry_priority, n)
    c9 = _affinity(affinity9, n)
    base = _ids(base_ids, n)
    if not isinstance(max_replace, Integral) or isinstance(max_replace, bool) or not base or max_replace < 0:
        raise ValueError("nonempty entry set and nonnegative cap required")
    b = min(int(max_replace), len(base), n - len(base))
    d = disagreement(g9, late)
    ranked_existing = sorted((i for i in base if d[i] > 0),
                             key=lambda i: (-d[i], i))
    reserved_donors = set(ranked_existing[:b])
    donors = sorted((i for i in base if i not in reserved_donors),
                    key=lambda i: (priority[i], i))
    outside = sorted(set(range(n)) - set(base))
    novelty = np.zeros(n)
    if outside:
        novelty[outside] = 1.0 - c9[np.ix_(outside, base)].max(axis=1)
    proposal = d * novelty
    candidates = sorted((i for i in outside if proposal[i] > 1e-12),
                        key=lambda i: (-proposal[i], i))
    used = min(b, len(donors), len(candidates))
    drop, add = donors[:used], candidates[:used]
    final = sorted((set(base) - set(drop)) | set(add))
    if len(final) != len(base):
        raise AssertionError("admission changed K")
    return {"ids": final, "added": add, "removed": drop,
            "actual_count": used, "disagreement": d.tolist()}


def matched_random_admission(base_ids, actual_plan, n, seed):
    """Same donors/count as the real admission; no label-dependent selection."""
    base = _ids(base_ids, n)
    drop = _ids(actual_plan["removed"], n)
    count = int(actual_plan["actual_count"])
    if count != len(drop) or not set(drop).issubset(base):
        raise ValueError("invalid matched donor plan")
    pool = sorted(set(range(n)) - set(base) - set(actual_plan["added"]))
    if len(pool) < count:
        raise ValueError("not enough distinct outside IDs for a matched control")
    rng = np.random.default_rng(seed)
    add = rng.choice(pool, size=count, replace=False).tolist() if count else []
    return {"ids": sorted((set(base) - set(drop)) | set(add)),
            "added": add, "removed": drop, "actual_count": count}


def initialize_anchors(entry_ids, risk, limit):
    d = _vector(risk)
    entry = _ids(entry_ids, len(d))
    if limit < 0 or (d < 0).any() or (d > 1).any():
        raise ValueError("invalid anchor state")
    chosen = sorted((i for i in entry if d[i] > 0),
                    key=lambda i: (-d[i], i))[:int(limit)]
    return {"ids": chosen, "weights": {str(i): float(d[i]) for i in chosen},
            "capacity": int(limit)}


def shuffle_entry_risk(entry_ids, risk, seed):
    """Permute the same risk values among the same admitted entry IDs once.

    Freeze this shuffled state across all cuts for the control. It is not
    reshuffled after each cut, and its number/values of positive risks match.
    """
    d = _vector(risk).copy()
    entry = _ids(entry_ids, len(d))
    rng = np.random.default_rng(seed)
    values = d[entry].copy()
    rng.shuffle(values)
    d[entry] = values
    return d


def cut_exchange(alive_ids, native_keep_ids, native_scores, query_scores,
                 anchors, affinity9, affinity_late, weight=0.25,
                 condition_on_query=True):
    """At most min(b, floor(K/8)) same-K exchanges at the current real cut.

    A positive gain is only a gain of this local proxy, never a guarantee of
    correct answers. All scores/hidden states are fixed for this cut. Later
    cuts recompute real query scores and cannot revive previously deleted IDs.
    """
    raw = _vector(native_scores)
    n = len(raw)
    query = _vector(query_scores, n)
    c9, cl = _affinity(affinity9, n), _affinity(affinity_late, n)
    alive = _ids(alive_ids, n)
    keep = _ids(native_keep_ids, n)
    if not keep or not set(keep).issubset(alive) or weight < 0:
        raise ValueError("invalid native cut")
    k = len(keep)
    b = int(anchors["capacity"])
    if b < 0:
        raise ValueError("invalid anchor capacity")
    anchor_ids = _ids(anchors["ids"], n)
    if len(anchor_ids) > b:
        raise ValueError("anchor count exceeds its declared capacity")
    source_weights = [float(anchors["weights"][str(i)]) for i in anchor_ids]
    if not np.isfinite(source_weights).all() or any(x < 0 or x > 1 for x in source_weights):
        raise ValueError("source risk must be finite in [0,1]")
    # An original anchor already absent from alive is diagnostic history only.
    # It cannot supply a current query score or be reinserted.
    active = [i for i in anchor_ids if i in set(alive)]
    cap = min(b, k // 8)
    p = np.zeros(n)
    u = np.zeros(n)
    p[alive] = rank01(raw[alive])
    if not condition_on_query:
        u[alive] = 1.0
    elif np.ptp(query[alive]) > 1e-12:
        u[alive] = rank01(query[alive])
    c = np.minimum(c9, cl)
    w = np.array([float(anchors["weights"][str(i)]) * u[i] for i in active])
    if not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError("invalid anchor weight")

    def objective(ids):
        native = float(p[ids].mean())
        support = 0.0
        if active and b:
            support = float((w * c[np.ix_(active, ids)].max(axis=1)).sum()) / b
        return native + weight * support

    current = set(keep)
    initial = objective(keep)
    swaps = []
    for _ in range(cap if weight > 0 and np.any(w > 0) else 0):
        candidates = [i for i in active if i not in current]
        donors = sorted(current, key=lambda i: (p[i], i))[:max(4, 2 * cap)]
        before = objective(sorted(current))
        best = None
        best_gain = 1e-12
        for add in sorted(candidates):
            for drop in donors:
                trial = sorted((current - {drop}) | {add})
                gain = objective(trial) - before
                if gain > best_gain:
                    best_gain, best = gain, (add, drop)
        if best is None:
            break
        add, drop = best
        current.remove(drop)
        current.add(add)
        swaps.append({"added": add, "removed": drop,
                      "proxy_gain": float(best_gain)})
    result = [i for i in alive if i in current]
    if len(result) != k or not set(result).issubset(alive):
        raise AssertionError("cut violated cardinality or resurrection constraint")
    return {"ids": result, "swaps": swaps, "max_swaps": cap,
            "net_changed_count": len(set(result) - set(keep)),
            "active_anchor_count": len(active),
            "proxy_before": initial, "proxy_after": objective(result)}


def matched_random_cut(alive_ids, native_keep_ids, actual_plan, n, seed):
    """Single-cut, shared-prefix diagnostic with the same donors/swap count.

    This must not be replayed at a different prefix to resurrect absent IDs.
    It is an offline negative control, not an online deployment rule.
    """
    alive = _ids(alive_ids, n)
    native = _ids(native_keep_ids, n)
    if not set(native).issubset(alive):
        raise ValueError("native set is not alive")
    final = _ids(actual_plan["ids"], n)
    if len(final) != len(native) or not set(final).issubset(alive):
        raise ValueError("control requires the same native prefix and capacity")
    drops = sorted(set(native) - set(final))
    real_adds = sorted(set(final) - set(native))
    pool = sorted(set(alive) - set(native) - set(real_adds))
    if len(pool) < len(drops):
        raise ValueError("not enough matched random alternatives")
    rng = np.random.default_rng(seed)
    add = rng.choice(pool, len(drops), replace=False).tolist() if drops else []
    chosen = (set(native) - set(drops)) | set(add)
    return {"ids": [i for i in alive if i in chosen], "added": add,
            "removed": drops, "actual_count": len(drops)}
