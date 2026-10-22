"""CPU reference diagnostics for existing visual evidence and query-attention drift.

These are ANALYSIS signals, not claims of causal utility or Shannon information.
The online Torch equivalent is in stage1_torch_adapter.py. Do not sort original
patch IDs before computing COVER *prefix* quality: that destroys greedy order.
"""
from __future__ import annotations
import numpy as np

PROBE_K = (16, 32, 64, 96)
PROBE_WEIGHT = (0.15, 0.35, 0.35, 0.15)

def coverage_difficulty(features, greedy_order, probe_k=PROBE_K,
                        weights=PROBE_WEIGHT):
    """Estimate remaining visual *feature coverage* along an existing greedy order.

    REQUIRE `greedy_order` from actual ordered COVER additions; do NOT pass
    spatially sorted selected IDs.  Formula r(k) = mean_i [1-max_{j in S_k}
    max(0, cosine(f_i,f_j))]. Higher r signals slower feature-space coverage.

    The full population remains all original patches, rather than only the
    currently retained candidates. Query relevance is NOT inferred.
    """
    x = np.asarray(features, dtype=np.float64)
    order = np.asarray(greedy_order)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must have shape [N,D]")
    if not np.isfinite(x).all():
        raise ValueError("non-finite vision features")
    if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("greedy_order must contain integer original patch IDs")
    if len(set(map(int, order))) != len(order):
        raise ValueError("duplicated greedy patch IDs")
    if (order < 0).any() or (order >= x.shape[0]).any():
        raise ValueError("greedy IDs outside original patch range")
    ks = tuple(int(k) for k in probe_k)
    if len(ks) != len(weights) or not ks or min(ks) <= 0:
        raise ValueError("probe_k and weights must be positive and matching")
    if max(ks) > len(order):
        raise ValueError("COVER greedy prefix too short; never pad with sorted IDs")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid nonnegative probe weights")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= 1e-12).any():
        raise ValueError("zero/degenerate visual vectors require explicit missing-feature handling")
    x = x / norm
    similarity = np.maximum(0.0, x @ x[order[:max(ks)]].T)
    running = np.maximum.accumulate(similarity, axis=1)
    residual = {
        k: float(np.maximum(0.0, 1.0-running[:, k-1]).mean())
        for k in ks
    }
    score = float(np.average([residual[k] for k in ks], weights=weights))
    return {"difficulty": score, "residual": residual,
            "metric": "cosine_feature_coverage_residual"}

def alignment_drift(original_ids, attention_now, attention_before, top_fraction=0.25):
    """Rank-set change on SAME original IDs; no visual-only softmax normalization.

    Args are raw nonnegative question-to-visual attention scores (or an
    equivalent score with frozen definition). Same IDs and same question mask
    MUST be used before interpreting drift. Intended at early boundary.
    """
    ids = list(map(int, original_ids))
    cur = np.asarray(attention_now, dtype=float)
    old = np.asarray(attention_before, dtype=float)
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError("unique nonempty patch ID list required")
    if cur.shape != (len(ids),) or old.shape != cur.shape:
        raise ValueError("current/history scores must align to same original IDs")
    if not np.isfinite(cur).all() or not np.isfinite(old).all():
        raise ValueError("non-finite attention signals")
    if (cur < 0).any() or (old < 0).any():
        raise ValueError("expected raw nonnegative attention")
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in (0,1]")
    if cur.sum() <= 0 or old.sum() <= 0:
        return {"valid": False, "churn": None, "reason": "zero_attention_mass"}
    k = max(1, round(len(ids) * top_fraction))
    # stable tie-breaking by ORIGINAL identity, not the current sequence order
    current = set(ids[i] for i in np.lexsort((ids, -cur))[:k])
    previous = set(ids[i] for i in np.lexsort((ids, -old))[:k])
    union = current | previous
    churn = 1.0 - len(current & previous) / len(union)
    return {"valid": True, "churn": float(churn), "top_k": k,
            "reason": "ok"}


def coverage_difficulty_from_sets(feature_layers, selected_sets, probe_k=PROBE_K,
                                  weights=PROBE_WEIGHT):
    """Assess ACTUAL COVER K-sets, not assumed greedy order.

    Important for old E014 helpers that return spatially sorted final subsets,
    or produce non-nested choices at different K. Each K-set must be a real
    output of the existing selector for that K. `feature_layers` is a list of
    raw Layer9 and penultimate patch matrices; both are normalized separately,
    then similarities are averaged (fixed, equal diagnostic weights).
    The *selection algorithm is unchanged*.
    """
    matrices = [np.asarray(f, dtype=np.float64) for f in feature_layers]
    if not matrices:
        raise ValueError("At least one visual feature layer required")
    n = matrices[0].shape[0]
    normalized = []
    for features in matrices:
        if features.ndim != 2 or features.shape[0] != n or n == 0 or features.shape[1] == 0:
            raise ValueError("All visual feature layers must share nonempty N patches")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite visual feature vector")
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError("Zero visual feature vector")
        normalized.append(features / norms)
    ks = tuple(int(k) for k in probe_k)
    if len(ks) != len(weights) or not ks or min(ks) <= 0:
        raise ValueError("Invalid probe dimensions")
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid probe weights")
    residual = {}
    for k in ks:
        if k not in selected_sets:
            raise ValueError(f"No actual K={k} COVER selection: cannot estimate it")
        raw = np.asarray(selected_sets[k])
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Selected original IDs must be integer 1D vectors")
        ids = list(map(int, raw))
        if len(ids) != k or len(set(ids)) != k or min(ids) < 0 or max(ids) >= n:
            raise ValueError(f"Invalid size/identity of actual K={k} selection")
        similarities = [
            np.maximum(0.0, f @ f[ids].T) for f in normalized
        ]
        joint_sim = np.mean(similarities, axis=0)
        residual[k] = float(np.maximum(0, 1 - joint_sim.max(axis=-1)).mean())
    score = float(np.average([residual[k] for k in ks], weights=w))
    return {"difficulty": score, "residual": residual,
            "metric": "actual_selected_sets_dual_layer_coverage_residual"}
