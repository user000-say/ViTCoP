"""Explicit cheap FINAL-event controls. They do not impersonate QED/RDRIFT."""
import numpy as np


def pooled_qk(query, visual_key, reader_weights=None, causal_bounds=None):
    """Mean raw scaled QK logits, not mean softmax attention.

    Q=[heads, original_question_readers, head_dim], K=[heads,N,head_dim].
    Both must use original RoPE, scaling convention and matched GQA heads.
    No generated answer or future state is allowed as an input.
    """
    if causal_bounds is None or len(causal_bounds) != 2 or not all(isinstance(x, (int, np.integer)) for x in causal_bounds):
        raise ValueError("actual (minimum reader position, maximum visual position) required")
    if not 0 <= causal_bounds[1] < causal_bounds[0]:
        raise ValueError("pooled reader must see EVERY visual key under the causal mask")
    q, k = np.asarray(query, float), np.asarray(visual_key, float)
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("expected matched [H,R,D] and [H,N,D]")
    if min(q.shape) == 0 or min(k.shape) == 0 or not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("empty/nonfinite QK")
    w = np.ones(q.shape[1]) if reader_weights is None else np.asarray(reader_weights, float)
    if w.shape != (q.shape[1],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid reader weights")
    pooled = np.einsum("hrd,r->hd", q, w / w.sum())
    return np.einsum("hd,hnd->n", pooled, k) / (q.shape[0] * np.sqrt(q.shape[2]))


def key_energy(visual_key):
    """Squared key L2 norm, a query-agnostic control."""
    k = np.asarray(visual_key, float)
    if k.ndim != 3 or min(k.shape) == 0 or not np.isfinite(k).all():
        raise ValueError("expected finite [H,N,D]")
    return np.square(k).sum(axis=(0, 2))


def topk(scores, keep):
    s = np.asarray(scores, float)
    if s.ndim != 1 or not np.isfinite(s).all() or not isinstance(keep, int) or not 0 < keep <= len(s):
        raise ValueError("invalid scores or K")
    return np.sort(np.argsort(-s, kind="stable")[:keep])


def random_keep(tokens, keep, seed):
    if not isinstance(tokens, int) or not isinstance(keep, int) or not 0 < keep <= tokens:
        raise ValueError("invalid token count")
    return np.sort(np.random.default_rng(seed).choice(tokens, keep, replace=False))
