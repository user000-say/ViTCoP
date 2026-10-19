"""Explicit new control: protect salient patches, fill by residual coverage.

This is NOT the historical E013 COVER. It uses ordinary facility-location
coverage and makes no algorithmic novelty or accuracy claim. All candidates
and demand patches remain eligible; raw pen payload is never modified.
"""
import numpy as np


POLICY = "protected_positive_cosine_half_uniform_half_saliency_v1"


def _dimensions(features, scores, k, slots):
    if (len(features.shape) != 2 or min(features.shape) == 0
            or scores.shape != (features.shape[0],)
            or type(k) is not int or not 0 < k <= len(scores)
            or type(slots) is not int or not 0 <= slots <= k):
        raise ValueError("aligned [N,D] / [N] and 0 <= coverage slots <= K required")


def numpy_keep(features, scores, k, slots):
    """CPU reference; return sorted local rows, with >= K-slots core retained."""
    x = np.asarray(features, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    _dimensions(x, s, k, slots)
    if not np.isfinite(x).all() or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("finite features and nonnegative saliency required")
    order = np.argsort(-s, kind="stable")
    if slots == 0:
        return np.sort(order[:k]).astype(np.int64)
    # Ranking candidates first makes an exact gain tie favor higher saliency.
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), np.float32(1e-12))
    affinity = np.maximum(z @ z[order].T, np.float32(0.))
    n = len(s)
    # Scaling before summation avoids overflow for large but finite scores.
    relative = s / max(float(s.max()), 1e-30)
    saliency_weights = relative / max(float(relative.sum()), 1e-30)
    if not relative.any():
        saliency_weights = np.full(n, 1/n, dtype=np.float32)
    weights = .5 * (saliency_weights + np.float32(1/n))
    core_size = k-slots
    selected = np.zeros(n, dtype=bool)
    selected[:core_size] = True
    best = affinity[:, :core_size].max(1) if core_size else np.zeros(n, dtype=np.float32)
    for _ in range(slots):
        gains = (weights[:, None] * np.maximum(affinity-best[:, None], 0)).sum(0)
        gains[selected] = -np.inf
        chosen = int(np.argmax(gains))
        selected[chosen] = True
        best = np.maximum(best, affinity[:, chosen])
    return np.sort(order[selected]).astype(np.int64)


def torch_keep(features, scores, k, slots, *, debug=True):
    """CUDA-resident implementation; no .cpu()/.numpy()/.item() in greedy loop.

    FP32 reductions and affinity are used even for BF16/FP16 payloads. Server
    preflight must check NumPy parity and actual input layout/device first.
    """
    import torch
    import torch.nn.functional as F
    _dimensions(features, scores, k, slots)
    if not features.is_floating_point() or not scores.is_floating_point() or features.device != scores.device:
        raise ValueError("floating device-aligned features and saliency required")
    if debug and (not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(scores).all())
                  or bool((scores < 0).any())):
        raise ValueError("finite features and nonnegative saliency required")
    with torch.no_grad():
        s = scores.float()
        order = torch.argsort(s, descending=True, stable=True)
        if slots == 0:
            return order[:k].sort().values
        z = F.normalize(features.float(), dim=-1, eps=1e-12)
        affinity = (z @ z.index_select(0, order).T).clamp_min_(0)
        n = len(s)
        relative = s/s.max().clamp_min(1e-30)
        normalized = relative/relative.sum().clamp_min(1e-30)
        uniform = torch.full_like(s, 1/n)
        saliency_weights = torch.where(relative.sum() > 0, normalized, uniform)
        weights = .5*(saliency_weights+uniform)
        core_size = k-slots
        selected = torch.zeros(n, dtype=torch.bool, device=s.device)
        selected[:core_size] = True
        best = affinity[:, :core_size].amax(1) if core_size else torch.zeros_like(s)
        for _ in range(slots):
            gains = ((affinity-best[:, None]).clamp_min(0)*weights[:, None]).sum(0)
            gains = gains.masked_fill(selected, -torch.inf)
            chosen = gains.argmax().reshape(1)
            selected.scatter_(0, chosen, True)
            best = torch.maximum(best, affinity.index_select(1, chosen).squeeze(1))
        # Boolean nonzero output length is fixed by K; no Python scalar read.
        return order[selected].sort().values


def callback(*, metric_features, scores, patch_ids, k, context):
    """Source-bound standard COVER callback, separate implementation family."""
    if context.get("coverage_policy") != POLICY:
        raise ValueError("explicit protected coverage policy required")
    slots = context["replacement_slots"]
    if len(patch_ids) != len(scores):
        raise ValueError("full aligned original patch IDs required")
    if isinstance(metric_features, np.ndarray):
        return numpy_keep(metric_features, scores, k, slots)
    return torch_keep(metric_features, scores, k, slots)


def make_spec():
    """Executable implementation binding; preflight flag deliberately false."""
    from pathlib import Path
    import hashlib
    source = Path(__file__).resolve()
    return dict(family="protected_coverage", entrypoint="elva.low_budget_cover:callback",
                supported_k=[60], replacement_slots={"60":12}, allowed_slots=[6,12,18],
                coverage_policy=POLICY, source_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                implementation_parity_verified=False)


def diagnostics(features, scores, keep, k, slots):
    """Offline numeric audit; caller exports real tensors OUTSIDE TTFT timing.

    Coverage is feature-space coverage, not a guarantee of OCR/region evidence.
    Spatial plots require actual crop coordinates in visual_audit separately.
    """
    from .evidence import indices
    x = np.asarray(features, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    _dimensions(x, s, k, slots)
    keep = indices(keep, len(s), k)
    if not np.isfinite(x).all() or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("finite nonnegative diagnostic evidence required")
    order = np.argsort(-s, kind="stable")
    baseline, core = order[:k], order[:k-slots]
    if not set(core).issubset(set(keep)):
        raise ValueError("protected salient core was not retained")
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    relative = s/max(float(s.max()),1e-30)
    norm = relative/max(float(relative.sum()),1e-30)
    if not relative.any():
        norm = np.full(len(s),1/len(s),dtype=np.float32)
    weights = .5*(norm+1/len(s))
    def coverage(rows):
        best = np.maximum(z @ z[rows].T, 0).max(1)
        return dict(uniform=float(best.mean()), weighted=float(weights @ best))
    added = sorted(set(map(int, keep))-set(map(int, baseline)))
    removed = sorted(set(map(int, baseline))-set(map(int, keep)))
    denominator = float(relative.sum())
    return dict(policy=POLICY, candidate_count=len(s), protected_count=len(core), slots=slots,
                actual_changed=len(added), added_local=added, removed_local=removed,
                topk_mass=float(relative[baseline].sum()/denominator) if denominator else None,
                kept_mass=float(relative[keep].sum()/denominator) if denominator else None,
                topk_coverage=coverage(baseline), kept_coverage=coverage(keep),
                zero_saliency=not bool(relative.any()))
