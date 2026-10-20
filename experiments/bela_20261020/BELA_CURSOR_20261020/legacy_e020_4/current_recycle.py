"""Current-event rectangular recycling; NumPy reference, not a VLM result.

Retains selected identities and order. Does not preserve future attention.
metric must be pre-existing CURRENT features aligned to active visual tokens.
"""
import numpy as np


def recycle_current(hidden, keep, metric=None, coords=None, groups=None,
                    alpha=.5, cosine_floor=.8, radius=2., positions=None,
                    prefix_only=False):
    x, k = np.asarray(hidden), np.asarray(keep)
    if x.ndim != 2 or min(x.shape) == 0 or x.dtype.kind != 'f':
        raise ValueError('floating nonempty hidden [N,D] required')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite hidden')
    n = len(x)
    if (k.ndim != 1 or k.dtype.kind not in 'iu' or not 0 < len(k) <= n
            or len(np.unique(k)) != len(k) or k.min() < 0 or k.max() >= n):
        raise ValueError('unique active-local keep indices required')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0,1]')
    base = x[k].copy()
    if alpha == 0 or len(k) == n:
        return base, {'off': True}
    if not np.isfinite(cosine_floor) or not -1 <= cosine_floor <= 1:
        raise ValueError('invalid cosine floor')
    f = np.asarray(metric, dtype=np.float64)
    if f.ndim != 2 or len(f) != n or f.shape[1] == 0 or not np.isfinite(f).all():
        raise ValueError('finite current metric [N,d] required')
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    f = np.divide(f, norm, out=np.zeros_like(f), where=norm > 0)
    active = np.ones(n, dtype=bool)
    active[k] = False
    donors = np.flatnonzero(active)
    sim = np.clip(f[donors] @ f[k].T, -1., 1.)
    eligible = sim >= cosine_floor
    if radius is not None:
        xy = np.asarray(coords, dtype=np.float64)
        if xy.shape != (n, 2) or not np.isfinite(xy).all():
            raise ValueError('original finite patch coordinates required')
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('invalid radius')
        eligible &= ((xy[donors, None] - xy[k][None]) ** 2).sum(-1) <= radius**2
    # Mandatory image/crop identities prevent accidental cross-image merging.
    g = np.asarray(groups)
    if g.shape != (n,) or g.dtype.kind not in 'iu':
        raise ValueError('integer image/crop groups [N] required')
    eligible &= g[donors, None] == g[k][None]
    if prefix_only:
        p = np.asarray(positions)
        if p.shape != (n,) or p.dtype.kind not in 'iu' or (p < 0).any():
            raise ValueError('original nonnegative sequence positions required')
        eligible &= p[donors, None] <= p[k][None]
    receiver = np.where(eligible, sim, -np.inf).argmax(-1)
    valid = eligible.any(-1)
    mass = np.bincount(receiver, weights=valid.astype(float), minlength=len(k))
    total = np.zeros((len(k), x.shape[1]), dtype=np.float64)
    np.add.at(total, receiver, x[donors].astype(float) * valid[:, None])
    # Algebraically the same bounded uniform merge as the cached reference.
    delta = alpha * (total - mass[:, None] * base) / (1 + mass[:, None])
    out = (base.astype(float) + delta).astype(x.dtype)
    return out, dict(off=False, donors=donors,
                     receiver=np.where(valid, receiver, -1), donor_mass=mass,
                     assigned=int(valid.sum()), metric_pairs=int(sim.size))
