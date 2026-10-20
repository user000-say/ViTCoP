"""Device-only current-event implementation; debug checks synchronize.

No complete N x N affinity. Production assumes validated finite model tensors.
Original model owns position IDs, masks, hidden compaction and KV handling.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def recycle_current(hidden, keep, metric=None, coords=None, groups=None,
                    alpha=.5, cosine_floor=.8, radius=2., positions=None,
                    prefix_only=False, debug=False):
    if hidden.ndim != 2 or min(hidden.shape) == 0 or not hidden.is_floating_point():
        raise ValueError('floating hidden [N,D] required')
    n, width = hidden.shape
    if keep.ndim != 1 or keep.dtype != torch.long or keep.device != hidden.device:
        raise ValueError('on-device LongTensor keep required')
    if not 0 < len(keep) <= n or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('invalid K/alpha')
    if debug:
        if keep.unique().numel() != len(keep) or bool(((keep < 0) | (keep >= n)).any()):
            raise ValueError('invalid indices')
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError('nonfinite hidden')
    base = hidden[keep]
    if alpha == 0 or len(keep) == n:
        return base, {'off': True}
    if not math.isfinite(cosine_floor) or not -1 <= cosine_floor <= 1:
        raise ValueError('invalid cosine floor')
    if (metric is None or metric.ndim != 2 or len(metric) != n
            or metric.shape[1] == 0 or metric.device != hidden.device):
        raise ValueError('on-device current metric [N,d] required')
    if groups is None or groups.shape != (n,) or groups.device != hidden.device or groups.dtype != torch.long:
        raise ValueError('on-device LongTensor crop groups required')
    if debug and not bool(torch.isfinite(metric).all()):
        raise ValueError('nonfinite metric')
    alive = torch.zeros(n, device=hidden.device, dtype=torch.bool)
    alive[keep] = True
    # Static N-K output size avoids nonzero() device-host size synchronization.
    donors = torch.argsort(alive.to(torch.int8), stable=True)[:n-len(keep)]
    f = F.normalize(metric.float(), dim=-1, eps=1e-30)
    sim = (f[donors] @ f[keep].T).clamp(-1, 1)
    eligible = sim >= cosine_floor
    if radius is not None:
        if coords is None or coords.shape != (n, 2) or coords.device != hidden.device:
            raise ValueError('original on-device coordinates required')
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('invalid radius')
        if debug and not bool(torch.isfinite(coords).all()):
            raise ValueError('nonfinite coordinates')
        eligible &= (coords[donors, None].float()-coords[keep][None].float()).square().sum(-1) <= radius**2
    eligible &= groups[donors, None] == groups[keep][None]
    if prefix_only:
        if positions is None or positions.shape != (n,) or positions.device != hidden.device or positions.dtype != torch.long:
            raise ValueError('original on-device LongTensor positions required')
        if debug and bool((positions < 0).any()):
            raise ValueError('negative sequence positions')
        eligible &= positions[donors, None] <= positions[keep][None]
    receiver = sim.masked_fill(~eligible, -torch.inf).argmax(-1)
    valid = eligible.any(-1)
    mass = torch.zeros(len(keep), device=hidden.device, dtype=torch.float32)
    total = torch.zeros((len(keep), width), device=hidden.device, dtype=torch.float32)
    mass.index_add_(0, receiver, valid.float())
    total.index_add_(0, receiver, hidden[donors].float()*valid[:, None])
    out = base.float() + alpha*(total-mass[:, None]*base.float())/(1+mass[:, None])
    return out.to(hidden.dtype), dict(off=False, donors=donors.detach(),
        receiver=torch.where(valid, receiver, -1).detach(), donor_mass=mass.detach(),
        assigned=valid.sum().detach(), metric_pairs=sim.numel())
