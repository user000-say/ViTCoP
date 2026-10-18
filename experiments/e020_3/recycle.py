"""Bounded aggregation AFTER an existing selector; NumPy reference.

No VLM training, no new tokens, no token resurrection. Output vectors change;
anchor identities/positions stay those selected by the existing runtime.
This does NOT preserve future attention or guarantee task accuracy.
"""
from dataclasses import dataclass
import numpy as np


def cosine_cache(features):
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or min(f.shape) == 0 or not np.isfinite(f).all():
        raise ValueError("features must be finite [entry_tokens, channels]")
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    f = np.divide(f, norm, out=np.zeros_like(f), where=norm > 0)
    return np.clip(f @ f.T, -1., 1.)


@dataclass
class Result:
    hidden: np.ndarray
    keep: np.ndarray
    donors: np.ndarray
    receiver: np.ndarray
    donor_mass: np.ndarray
    anchor_mass: np.ndarray
    audit: dict


def recycle(hidden, keep, affinity=None, coords=None, groups=None, weights=None,
            alpha=.5, cosine_floor=.8, radius=2., assignment="semantic",
            preserve_norm=False, seed=0):
    """Aggregate eligible discarded states into selected states.

    keep is active-local and its original order is preserved. coords are
    original patch-grid coordinates; groups identify distinct crops/images.
    affinity is a cosine matrix from already-computed features, typically
    cached once on the Stage1 pool. Uniform weights avoid extra QK reads.
    """
    x0, k = np.asarray(hidden), np.asarray(keep)
    if x0.ndim != 2 or min(x0.shape) == 0 or x0.dtype.kind != "f" or not np.isfinite(x0).all():
        raise ValueError("hidden must be finite floating [N,D]")
    n = len(x0)
    if k.ndim != 1 or k.dtype.kind not in "iu" or not 0 < len(k) <= n:
        raise ValueError("keep must be a nonempty integer vector")
    if len(np.unique(k)) != len(k) or k.min() < 0 or k.max() >= n:
        raise ValueError("invalid/duplicate active-local keep index")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    alive = np.zeros(n, bool)
    alive[k] = True
    d = np.flatnonzero(~alive)
    # An exact off switch: no cache or geometry is read, no FP arithmetic.
    if alpha == 0 or len(d) == 0:
        return Result(x0[k].copy(), k.copy(), d, np.full(len(d), -1, int),
                      np.zeros(len(k)), np.ones(len(k)),
                      {"assigned": 0, "positive_weight_donors": 0, "off": True})
    if not np.isfinite(cosine_floor) or not -1 <= cosine_floor <= 1:
        raise ValueError("cosine_floor must lie in [-1,1]")
    if assignment not in ("semantic", "spatial", "random"):
        raise ValueError("unknown assignment")
    a = np.asarray(affinity, dtype=np.float64)
    if a.shape != (n, n) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1 + 1e-6:
        raise ValueError("affinity must be a finite cosine [N,N] matrix")
    sim = a[np.ix_(d, k)]
    eligible = sim >= cosine_floor
    distance = None
    if radius is not None or assignment == "spatial":
        xy = np.asarray(coords, dtype=np.float64)
        if xy.shape != (n, 2) or not np.isfinite(xy).all():
            raise ValueError("original [N,2] patch coordinates required")
        distance = np.square(xy[d, None] - xy[k][None]).sum(axis=-1)
    if radius is not None:
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be nonnegative, or None")
        eligible &= distance <= radius * radius
    if groups is not None:
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError("one crop/image group per active token required")
        eligible &= g[d, None] == g[k][None]
    if assignment == "semantic":
        score = sim
    elif assignment == "spatial":
        score = -distance
    else:
        score = np.random.default_rng(seed).random(sim.shape)
    receiver = np.where(eligible, score, -np.inf).argmax(axis=1)
    valid = eligible.any(axis=1)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and nonnegative; not raw QED scores")
    fallback = w.sum() == 0
    if fallback:
        w = np.ones(n)
    dw = w[d] * valid
    mass = np.zeros(len(k))
    total = np.zeros((len(k), x0.shape[1]))
    np.add.at(mass, receiver, dw)
    np.add.at(total, receiver, x0[d].astype(np.float64) * dw[:, None])
    center = np.divide(total, mass[:, None], out=np.zeros_like(total), where=mass[:, None] > 0)
    fraction = np.divide(alpha * mass, w[k] + mass,
                         out=np.zeros_like(mass), where=w[k] + mass > 0)
    anchors = x0[k].astype(np.float64)
    out = anchors + fraction[:, None] * (center - anchors)
    if preserve_norm:
        size = np.linalg.norm(out, axis=1, keepdims=True)
        original_size = np.linalg.norm(anchors, axis=1, keepdims=True)
        restored = out * original_size / np.maximum(size, 1e-12)
        out = np.where(size > 1e-12, restored, anchors)
    displacement = np.linalg.norm(out - anchors, axis=1)
    return Result(out.astype(x0.dtype), k.copy(), d, np.where(valid, receiver, -1),
                  mass, w[k], {"assigned": int(valid.sum()),
                  "positive_weight_donors": int((dw > 0).sum()), "off": False,
                  "uniform_weight_fallback": bool(fallback),
                  "changed_anchors": int((displacement > 1e-12).sum()),
                  "max_displacement": float(displacement.max(initial=0.))})


class AffinityCache:
    """Per-input cache, indexed by entry SLOT IDs rather than patch IDs."""
    def __init__(self, source_id, features, coords, groups=None):
        self.source_id = str(source_id)
        self.affinity = cosine_cache(features)
        self.coords = np.asarray(coords)
        self.groups = None if groups is None else np.asarray(groups)
        n = len(self.affinity)
        if self.coords.shape != (n, 2) or not np.isfinite(self.coords).all():
            raise ValueError("entry coordinates required")
        if self.groups is not None and self.groups.shape != (n,):
            raise ValueError("invalid entry groups")

    def view(self, source_id, active_slots):
        if str(source_id) != self.source_id:
            raise ValueError("cache belongs to another input")
        s = np.asarray(active_slots)
        if s.ndim != 1 or s.dtype.kind not in "iu" or len(s) == 0:
            raise ValueError("nonempty integer entry slots required")
        if len(np.unique(s)) != len(s) or s.min() < 0 or s.max() >= len(self.affinity):
            raise ValueError("invalid active entry slots")
        return dict(affinity=self.affinity[np.ix_(s, s)], coords=self.coords[s],
                    groups=None if self.groups is None else self.groups[s])
